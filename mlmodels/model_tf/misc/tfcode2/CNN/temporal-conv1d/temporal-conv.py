#!/usr/bin/env python
# coding: utf-8

# In[1]:


import functools
import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# In[2]:


mnist = input_data.read_data_sets("", one_hot=True)


# In[3]:


def residual_block(x, i, filters, kernel_size):
    x_copy = x
    pad_len = (kernel_size - 1) * i
    x = tf.pad(x, [[0, 0], [pad_len, 0], [0, 0]])
    x = tf.layers.conv1d(x, filters, kernel_size, dilation_rate=i, padding="valid")
    tanh = tf.nn.tanh(x)
    sigmoid = tf.nn.sigmoid(x)
    x = tanh * sigmoid
    x = tf.layers.dropout(x, 0.05, noise_shape=[x.shape[0], x.shape[1], tf.constant(1)])
    x = tf.layers.conv1d(x, filters, 1, padding="same")
    return x_copy + x, x


class Model:
    def __init__(self, filters=32, kernel_size=4, dilations=[1, 2, 4, 8], stacks=8):
        self.X = tf.placeholder(tf.float32, [None, 28, 28])
        self.Y = tf.placeholder(tf.float32, [None, 10])
        padded_x = tf.pad(self.X, [[0, 0], [(filters - 1), 0], [0, 0]])
        padded_x = tf.layers.conv1d(padded_x, filters, kernel_size, dilation_rate=1)
        for s in range(stacks):
            for i in dilations:
                padded_x, skip_out = residual_block(padded_x, i, filters, kernel_size)
        self.logits = tf.layers.dense(padded_x[:, -1], 10)
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


# In[4]:


sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())


# In[ ]:


EPOCH = 10
BATCH_SIZE = 128


# In[ ]:


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
