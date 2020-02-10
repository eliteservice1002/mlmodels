#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import tensorflow as tf
from sklearn.cross_validation import train_test_split

from utils import *

# In[2]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[3]:


ONEHOT = np.zeros((len(trainset.data), len(trainset.target_names)))
ONEHOT[np.arange(len(trainset.data)), trainset.target] = 1.0
train_X, test_X, train_Y, test_Y, train_onehot, test_onehot = train_test_split(
    trainset.data, trainset.target, ONEHOT, test_size=0.2
)


# In[4]:


concat = " ".join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print("vocab from size: %d" % (vocabulary_size))
print("Most common words", count[4:10])
print("Sample data", data[:10], [rev_dictionary[i] for i in data[:10]])


# In[5]:


GO = dictionary["GO"]
PAD = dictionary["PAD"]
EOS = dictionary["EOS"]
UNK = dictionary["UNK"]


# In[6]:


class NTMCell:
    def __init__(
        self,
        rnn_size,
        memory_size,
        memory_vector_dim,
        read_head_num,
        write_head_num,
        addressing_mode="content_and_location",
        shift_range=1,
        reuse=False,
        output_dim=None,
    ):
        self.rnn_size = rnn_size
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.addressing_mode = addressing_mode
        self.reuse = reuse
        self.controller = tf.nn.rnn_cell.BasicRNNCell(self.rnn_size)
        self.step = 0
        self.output_dim = output_dim
        self.shift_range = shift_range

    def __call__(self, x, prev_state):
        prev_read_vector_list = prev_state["read_vector_list"]
        prev_controller_state = prev_state["controller_state"]
        controller_input = tf.concat([x] + prev_read_vector_list, axis=1)
        with tf.variable_scope("controller", reuse=self.reuse):
            controller_output, controller_state = self.controller(
                controller_input, prev_controller_state
            )
        num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1
        num_heads = self.read_head_num + self.write_head_num
        total_parameter_num = (
            num_parameters_per_head * num_heads + self.memory_vector_dim * 2 * self.write_head_num
        )
        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            o2p_w = tf.get_variable(
                "o2p_w",
                [controller_output.get_shape()[1], total_parameter_num],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5),
            )
            o2p_b = tf.get_variable(
                "o2p_b",
                [total_parameter_num],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5),
            )
            parameters = tf.nn.xw_plus_b(controller_output, o2p_w, o2p_b)
        head_parameter_list = tf.split(
            parameters[:, : num_parameters_per_head * num_heads], num_heads, axis=1
        )
        erase_add_list = tf.split(
            parameters[:, num_parameters_per_head * num_heads :], 2 * self.write_head_num, axis=1
        )
        prev_w_list = prev_state["w_list"]
        prev_M = prev_state["M"]
        w_list = []
        p_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = tf.tanh(head_parameter[:, 0 : self.memory_vector_dim])
            beta = tf.sigmoid(head_parameter[:, self.memory_vector_dim]) * 10
            g = tf.sigmoid(head_parameter[:, self.memory_vector_dim + 1])
            s = tf.nn.softmax(
                head_parameter[
                    :,
                    self.memory_vector_dim
                    + 2 : self.memory_vector_dim
                    + 2
                    + (self.shift_range * 2 + 1),
                ]
            )
            gamma = tf.log(tf.exp(head_parameter[:, -1]) + 1) + 1
            with tf.variable_scope("addressing_head_%d" % i):
                w = self.addressing(k, beta, g, s, gamma, prev_M, prev_w_list[i])  # Figure 2
            w_list.append(w)
            p_list.append({"k": k, "beta": beta, "g": g, "s": s, "gamma": gamma})
        read_w_list = w_list[: self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], dim=2) * prev_M, axis=1)
            read_vector_list.append(read_vector)
        write_w_list = w_list[self.read_head_num :]
        M = prev_M
        for i in range(self.write_head_num):
            w = tf.expand_dims(write_w_list[i], axis=2)
            erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)
            add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1)
            M = M * (tf.ones(M.get_shape()) - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector)

        if not self.output_dim:
            output_dim = x.get_shape()[1]
        else:
            output_dim = self.output_dim
        with tf.variable_scope("o2o", reuse=(self.step > 0) or self.reuse):
            o2o_w = tf.get_variable(
                "o2o_w",
                [controller_output.get_shape()[1], output_dim],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5),
            )
            o2o_b = tf.get_variable(
                "o2o_b",
                [output_dim],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5),
            )
            NTM_output = tf.nn.xw_plus_b(controller_output, o2o_w, o2o_b)
        state = {
            "controller_state": controller_state,
            "read_vector_list": read_vector_list,
            "w_list": w_list,
            "p_list": p_list,
            "M": M,
        }
        self.step += 1
        return NTM_output, state

    def addressing(self, k, beta, g, s, gamma, prev_M, prev_w):
        k = tf.expand_dims(k, axis=2)
        inner_product = tf.matmul(prev_M, k)
        k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keep_dims=True))
        M_norm = tf.sqrt(tf.reduce_sum(tf.square(prev_M), axis=2, keep_dims=True))
        norm_product = M_norm * k_norm
        K = tf.squeeze(inner_product / (norm_product + 1e-8))
        K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K)
        w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keep_dims=True)
        if self.addressing_mode == "content":
            return w_c
        g = tf.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w

        s = tf.concat(
            [
                s[:, : self.shift_range + 1],
                tf.zeros([s.get_shape()[0], self.memory_size - (self.shift_range * 2 + 1)]),
                s[:, -self.shift_range :],
            ],
            axis=1,
        )
        t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
        s_matrix = tf.stack(
            [
                t[:, self.memory_size - i - 1 : self.memory_size * 2 - i - 1]
                for i in range(self.memory_size)
            ],
            axis=1,
        )
        w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)
        w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=1))
        w = w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keep_dims=True)
        return w

    def zero_state(self, batch_size, dtype):
        def expand(x, dim, N):
            return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

        with tf.variable_scope("init", reuse=self.reuse):
            state = {
                "controller_state": expand(
                    tf.tanh(
                        tf.get_variable(
                            "init_state",
                            self.rnn_size,
                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5),
                        )
                    ),
                    dim=0,
                    N=batch_size,
                ),
                "read_vector_list": [
                    expand(
                        tf.nn.softmax(
                            tf.get_variable(
                                "init_r_%d" % i,
                                [self.memory_vector_dim],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5),
                            )
                        ),
                        dim=0,
                        N=batch_size,
                    )
                    for i in range(self.read_head_num)
                ],
                "w_list": [
                    expand(
                        tf.nn.softmax(
                            tf.get_variable(
                                "init_w_%d" % i,
                                [self.memory_size],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5),
                            )
                        ),
                        dim=0,
                        N=batch_size,
                    )
                    if self.addressing_mode == "content_and_loaction"
                    else tf.zeros([batch_size, self.memory_size])
                    for i in range(self.read_head_num + self.write_head_num)
                ],
                "M": expand(
                    tf.tanh(
                        tf.get_variable(
                            "init_M",
                            [self.memory_size, self.memory_vector_dim],
                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5),
                        )
                    ),
                    dim=0,
                    N=batch_size,
                ),
            }
            return state


# In[7]:


class Model:
    def __init__(
        self,
        seq_len,
        size_layer,
        batch_size,
        dict_size,
        dimension_input,
        dimension_output,
        learning_rate,
        memory_size,
        memory_vector_size,
        read_head_num=4,
        write_head_num=1,
    ):
        self.X = tf.placeholder(tf.int32, [batch_size, seq_len])
        self.Y = tf.placeholder(tf.float32, [batch_size, dimension_output])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, dimension_input], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        cell = NTMCell(
            size_layer,
            memory_size,
            memory_vector_size,
            read_head_num=read_head_num,
            write_head_num=write_head_num,
            addressing_mode="content_and_location",
            output_dim=dimension_output,
        )
        state = cell.zero_state(batch_size, tf.float32)
        self.state_list = [state]
        self.o = []
        o2o_w = tf.Variable(tf.random_normal((dimension_output, dimension_output)))
        o2o_b = tf.Variable(tf.random_normal([dimension_output]))
        for t in range(seq_len):
            output, state = cell(encoder_embedded[:, t, :], state)
            output = tf.nn.xw_plus_b(output, o2o_w, o2o_b)
            self.o.append(output)
            self.state_list.append(state)
        self.o = tf.stack(self.o, axis=1)
        self.state_list.append(state)
        self.logits = self.o[:, -1]
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


# In[ ]:


size_layer = 128
embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 32
memory_size = 128
memory_vector_size = 40


# In[ ]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(
    maxlen,
    size_layer,
    batch_size,
    vocabulary_size + 4,
    embedded_size,
    dimension_output,
    learning_rate,
    memory_size,
    memory_vector_size,
)
sess.run(tf.global_variables_initializer())


# In[ ]:


EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:%d\n" % (EPOCH))
        break

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (len(train_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(train_X[i : i + batch_size], dictionary, maxlen)
        acc, loss, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X: batch_x, model.Y: train_onehot[i : i + batch_size]},
        )
        train_loss += loss
        train_acc += acc

    for i in range(0, (len(test_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(test_X[i : i + batch_size], dictionary, maxlen)
        acc, loss = sess.run(
            [model.accuracy, model.cost],
            feed_dict={model.X: batch_x, model.Y: test_onehot[i : i + batch_size]},
        )
        test_loss += loss
        test_acc += acc

    train_loss /= len(train_X) // batch_size
    train_acc /= len(train_X) // batch_size
    test_loss /= len(test_X) // batch_size
    test_acc /= len(test_X) // batch_size

    if test_acc > CURRENT_ACC:
        print("epoch: %d, pass acc: %f, current acc: %f" % (EPOCH, CURRENT_ACC, test_acc))
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
    else:
        CURRENT_CHECKPOINT += 1

    print("time taken:", time.time() - lasttime)
    print(
        "epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n"
        % (EPOCH, train_loss, train_acc, test_loss, test_acc)
    )
    EPOCH += 1
