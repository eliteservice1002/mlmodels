#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# In[2]:


mnist = input_data.read_data_sets("", one_hot=True)


# In[3]:


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
    def __init__(self, learning_rate=1e-4, top_k=5, n_filters=250):
        self.n_filters = n_filters
        self.kernels = [3, 4, 5]
        self.top_k = top_k
        self.X = tf.placeholder(tf.float32, [None, 28, 28])
        self.Y = tf.placeholder(tf.float32, [None, 10])
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
        self.logits = tf.layers.dense(parallels, 10)
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


# In[4]:


sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())


# In[5]:


EPOCH = 10
BATCH_SIZE = 128


# In[6]:


for i in range(EPOCH):
    last = time.time()
    TOTAL_LOSS, ACCURACY = 0, 0
    for n in range(0, (mnist.train.images.shape[0] // BATCH_SIZE) * BATCH_SIZE, BATCH_SIZE):
        batch_x = mnist.train.images[n : n + BATCH_SIZE, :].reshape((-1, 28, 28))
        acc, cost, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X: batch_x, model.Y: mnist.train.labels[n : n + BATCH_SIZE, :]},
        )
        ACCURACY += acc
        TOTAL_LOSS += cost
    TOTAL_LOSS /= mnist.train.images.shape[0] // BATCH_SIZE
    ACCURACY /= mnist.train.images.shape[0] // BATCH_SIZE
    print(
        "epoch %d, avg loss %f, avg acc %f, time taken %f secs"
        % (i + 1, TOTAL_LOSS, ACCURACY, time.time() - last)
    )


# In[ ]:
