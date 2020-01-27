#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import os
import pickle
import random
import re
import time

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

# In[2]:


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


# In[3]:


class Model:
    def __init__(
        self,
        seq_len,
        size_layer,
        batch_size,
        dimension_input,
        dimension_output,
        learning_rate,
        memory_size,
        memory_vector_size,
        read_head_num=4,
        write_head_num=1,
    ):
        self.X = tf.placeholder(tf.float32, [batch_size, seq_len, dimension_input])
        self.Y = tf.placeholder(tf.float32, [batch_size, dimension_output])
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
            output, state = cell(self.X[:, t, :], state)
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


# In[4]:


# In[5]:


maxlen = 20
size_layer = 256
learning_rate = 0.0001
batch = 100
memory_size = 128
memory_vector_size = 40


# In[6]:


with open("dataset-emotion.p", "rb") as fopen:
    df = pickle.load(fopen)
with open("vector-emotion.p", "rb") as fopen:
    vectors = pickle.load(fopen)
with open("dataset-dictionary.p", "rb") as fopen:
    dictionary = pickle.load(fopen)


# In[7]:


label = np.unique(df[:, 1])


# In[ ]:


train_X, test_X, train_Y, test_Y = train_test_split(df[:, 0], df[:, 1].astype("int"), test_size=0.2)


# In[ ]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(
    maxlen,
    size_layer,
    batch,
    vectors.shape[1],
    label.shape[0],
    learning_rate,
    memory_size,
    memory_vector_size,
)
sess.run(tf.global_variables_initializer())
dimension = vectors.shape[1]
saver = tf.train.Saver(tf.global_variables())
EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 10, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:", EPOCH)
        break
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (train_X.shape[0] // batch) * batch, batch):
        batch_x = np.zeros((batch, maxlen, dimension))
        batch_y = np.zeros((batch, len(label)))
        for k in range(batch):
            tokens = train_X[i + k].split()[:maxlen]
            emb_data = np.zeros((maxlen, dimension), dtype=np.float32)
            for no, text in enumerate(tokens[::-1]):
                try:
                    emb_data[-1 - no, :] += vectors[dictionary[text], :]
                except Exception as e:
                    print(e)
                    continue
            batch_y[k, int(train_Y[i + k])] = 1.0
            batch_x[k, :, :] = emb_data[:, :]
        loss, _ = sess.run(
            [model.cost, model.optimizer], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        train_loss += loss
        train_acc += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})

    for i in range(0, (test_X.shape[0] // batch) * batch, batch):
        batch_x = np.zeros((batch, maxlen, dimension))
        batch_y = np.zeros((batch, len(label)))
        for k in range(batch):
            tokens = test_X[i + k].split()[:maxlen]
            emb_data = np.zeros((maxlen, dimension), dtype=np.float32)
            for no, text in enumerate(tokens[::-1]):
                try:
                    emb_data[-1 - no, :] += vectors[dictionary[text], :]
                except:
                    continue
            batch_y[k, int(test_Y[i + k])] = 1.0
            batch_x[k, :, :] = emb_data[:, :]
        loss, acc = sess.run(
            [model.cost, model.accuracy], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        test_loss += loss
        test_acc += acc

    train_loss /= train_X.shape[0] // batch
    train_acc /= train_X.shape[0] // batch
    test_loss /= test_X.shape[0] // batch
    test_acc /= test_X.shape[0] // batch
    if test_acc > CURRENT_ACC:
        print("epoch:", EPOCH, ", pass acc:", CURRENT_ACC, ", current acc:", test_acc)
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
        saver.save(sess, os.getcwd() + "/model-rnn-vector.ckpt")
    else:
        CURRENT_CHECKPOINT += 1
    EPOCH += 1
    print("time taken:", time.time() - lasttime)
    print(
        "epoch:",
        EPOCH,
        ", training loss:",
        train_loss,
        ", training acc:",
        train_acc,
        ", valid loss:",
        test_loss,
        ", valid acc:",
        test_acc,
    )


# In[ ]:
