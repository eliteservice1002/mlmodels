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

# In[2]:


maxlen = 20
location = os.getcwd()
learning_rate = 0.0001
batch = 100


# In[3]:


with open("dataset-emotion.p", "rb") as fopen:
    df = pickle.load(fopen)
with open("vector-emotion.p", "rb") as fopen:
    vectors = pickle.load(fopen)
with open("dataset-dictionary.p", "rb") as fopen:
    dictionary = pickle.load(fopen)


# In[4]:


label = np.unique(df[:, 1])

train_X, test_X, train_Y, test_Y = train_test_split(df[:, 0], df[:, 1].astype("int"), test_size=0.2)


# In[7]:


def add_conv1d(x, n_filters, kernel_size, strides=1):
    return tf.layers.conv1d(
        inputs=x,
        filters=n_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="valid",
        use_bias=True,
        activation=tf.nn.relu,
    )


class Model:
    def __init__(
        self, seq_len, dimension_input, dimension_output, learning_rate, top_k=5, n_filters=250
    ):
        self.n_filters = n_filters
        self.kernels = [3, 4, 5]
        self.top_k = top_k
        self.X = tf.placeholder(tf.float32, [None, seq_len, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        parallels = []
        for k in self.kernels:
            p = add_conv1d(self.X, self.n_filters // len(self.kernels), kernel_size=k)
            p = self.add_kmax_pooling(p)
            parallels.append(p)
        parallels = tf.concat(parallels, axis=-1)
        parallels = tf.reshape(
            parallels,
            [-1, self.top_k * (len(self.kernels) * (self.n_filters // len(self.kernels)))],
        )
        feed = tf.nn.dropout(tf.layers.dense(parallels, self.n_filters, tf.nn.relu), 0.5)
        self.logits = tf.layers.dense(parallels, dimension_output)
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def add_kmax_pooling(self, x):
        Y = tf.transpose(x, [0, 2, 1])
        Y = tf.nn.top_k(Y, self.top_k, sorted=False).values
        Y = tf.transpose(Y, [0, 2, 1])
        return tf.reshape(Y, [-1, self.top_k, self.n_filters // len(self.kernels)])


# In[8]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(maxlen, vectors.shape[1], label.shape[0], learning_rate)
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
