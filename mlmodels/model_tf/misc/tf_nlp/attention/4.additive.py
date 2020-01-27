#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import numpy as np
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


# In[9]:


class Model:
    def __init__(
        self,
        size_layer,
        num_layers,
        embedded_size,
        dict_size,
        dimension_output,
        learning_rate,
        maxlen,
    ):
        def cells(reuse=False):
            return tf.nn.rnn_cell.GRUCell(size_layer, reuse=reuse)

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        outputs, _ = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded, dtype=tf.float32)

        U = tf.get_variable(
            "U",
            shape=[size_layer, embedded_size],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
        )
        V = tf.get_variable(
            "V",
            shape=[embedded_size, embedded_size],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
        )
        Av = tf.get_variable(
            "Av",
            shape=[embedded_size, 1],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
        )

        timesteps = tf.shape(outputs)[1]

        ref_proj = tf.reshape(
            tf.matmul(tf.reshape(outputs, [-1, size_layer]), U), [-1, timesteps, embedded_size]
        )
        hi = tf.expand_dims(tf.matmul(encoder_embedded[:, -1], V), axis=1)
        blended = ref_proj + hi
        self.scores = tf.reshape(
            tf.matmul(tf.reshape(blended, [-1, embedded_size]), Av), [-1, timesteps]
        )
        W = tf.get_variable(
            "w", shape=(maxlen, dimension_output), initializer=tf.orthogonal_initializer()
        )
        b = tf.get_variable("b", shape=(dimension_output), initializer=tf.zeros_initializer())
        self.logits = tf.matmul(self.scores, W) + b
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[10]:


size_layer = 128
num_layers = 2
embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 128
tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(
    size_layer, num_layers, embedded_size, len(dictionary), dimension_output, learning_rate, maxlen
)
sess.run(tf.global_variables_initializer())


# In[11]:


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


# In[12]:


logits = sess.run(model.logits, feed_dict={model.X: str_idx(test_X, dictionary, maxlen)})
print(
    metrics.classification_report(test_Y, np.argmax(logits, 1), target_names=trainset.target_names)
)


# In[ ]:
