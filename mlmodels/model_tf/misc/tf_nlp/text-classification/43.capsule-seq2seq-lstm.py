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


def squash(X, epsilon=1e-9):
    vec_squared_norm = tf.reduce_sum(tf.square(X), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    return scalar_factor * X


def conv_layer(X, num_output, num_vector, kernel=None, stride=None):
    global batch_size
    capsules = tf.layers.conv1d(
        X, num_output * num_vector, kernel, stride, padding="VALID", activation=tf.nn.relu
    )
    capsules = tf.reshape(capsules, (batch_size, -1, num_vector, 1))
    return squash(capsules)


def routing(X, b_IJ, seq_len, dimension_out, routing_times=2):
    global batch_size
    shape_X = X.shape[1].value
    w = tf.Variable(tf.truncated_normal([1, shape_X, seq_len, 8, dimension_out // 2], stddev=1e-1))
    X = tf.tile(X, [1, 1, seq_len, 1, dimension_out])
    w = tf.tile(w, [batch_size, 1, 1, 1, routing_times])
    print("X shape: %s, w shape: %s" % (str(X.shape), str(w.shape)))
    u_hat = tf.matmul(w, X, transpose_a=True)
    u_hat_stopped = tf.stop_gradient(u_hat)
    for i in range(routing_times):
        c_IJ = tf.nn.softmax(b_IJ, dim=2)
        if i == routing_times - 1:
            s_J = tf.multiply(c_IJ, u_hat)
            s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
            v_J = squash(s_J)
        else:
            s_J = tf.multiply(c_IJ, u_hat_stopped)
            s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
            v_J = squash(s_J)
            v_J_tiled = tf.tile(v_J, [1, shape_X, 1, 1, 1])
            u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
            b_IJ += u_produce_v
    return v_J


def fully_conn_layer(X, num_output, dimension_out):
    global batch_size
    X_ = tf.reshape(X, shape=(batch_size, -1, 1, X.shape[-2].value, 1))
    b_IJ = tf.constant(np.zeros([batch_size, X.shape[1].value, num_output, 1, 1], dtype=np.float32))
    capsules = routing(X_, b_IJ, num_output, dimension_out, routing_times=2)
    capsules = tf.squeeze(capsules, axis=1)
    return capsules


class CapsuleNetwork:
    def __init__(
        self,
        batch_size,
        learning_rate,
        seq_len,
        size_layer,
        num_layers,
        maxlen,
        dict_size,
        embedded_size,
        dimension_output,
        kernels=[6, 3, 2],
        strides=[3, 2, 1],
        epsilon=1e-8,
        skip=5,
    ):
        def cells(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(
                size_layer, initializer=tf.orthogonal_initializer(), reuse=reuse
            )

        self.X = tf.placeholder(tf.int32, [batch_size, maxlen])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)

        results = []
        for i in range(len(kernels)):
            conv = tf.layers.conv1d(
                encoder_embedded,
                filters=32,
                kernel_size=kernels[i],
                strides=strides[i],
                padding="VALID",
            )
            caps1 = conv_layer(conv, 8, 8, kernels[i], strides[i])
            caps2 = fully_conn_layer(caps1, seq_len, 32)
            v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True) + epsilon)[
                :, :, 0, :
            ]
            print("output shape: %s" % (str(v_length.shape)))
            results.append(v_length)
        results = tf.concat(results, 1)
        decoder_embedded = results[:, skip:, :]

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        _, last_state = tf.nn.dynamic_rnn(rnn_cells, results, dtype=tf.float32)

        with tf.variable_scope("decoder"):
            rnn_cells_dec = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
            outputs, _ = tf.nn.dynamic_rnn(
                rnn_cells_dec, decoder_embedded, initial_state=last_state, dtype=tf.float32
            )

        W = tf.get_variable(
            "w", shape=(size_layer, dimension_output), initializer=tf.orthogonal_initializer()
        )
        b = tf.get_variable("b", shape=(dimension_output), initializer=tf.zeros_initializer())
        self.logits = tf.matmul(outputs[:, -1], W) + b
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[7]:


size_layer = 128
maxlen = 50
num_layers = 2
embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-4
batch_size = 64


# In[ ]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = CapsuleNetwork(
    batch_size,
    learning_rate,
    5,
    size_layer,
    num_layers,
    maxlen,
    vocabulary_size + 4,
    embedded_size,
    dimension_output,
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


# In[ ]:
