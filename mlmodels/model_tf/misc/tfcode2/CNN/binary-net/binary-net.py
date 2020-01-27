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


def activation(x):
    x = tf.clip_by_value(x, -1.0, 1.0)
    return x + tf.stop_gradient(tf.sign(x) - x)


def weight_bias(shape):
    init = tf.random_uniform(shape, -1.0, 1.0)
    x, y = tf.Variable(init), tf.Variable(init)
    coeff = np.float32(1.0 / np.sqrt(1.5 / (np.prod(shape[:-2]) * (shape[-2] + shape[-1]))))
    tmp = y + coeff * (x - y)
    tmp = tf.clip_by_value(tmp, -1.0, 1.0)
    tmp = tf.group(x.assign(tmp), y.assign(tmp))
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tmp)
    x = tf.clip_by_value(x, -1.0, 1.0)
    xbin = tf.sign(x) * tf.reduce_mean(tf.abs(x), axis=[0, 1, 2])
    x = x + tf.stop_gradient(xbin - x)
    return x, tf.Variable(tf.constant(0.1, shape=[shape[-1]]))


def batch_norm(x, epsilon, decay=0.9, is_training=True):
    return tf.contrib.layers.batch_norm(
        x,
        decay=decay,
        center=True,
        scale=True,
        epsilon=epsilon,
        updates_collections=None,
        is_training=is_training,
        trainable=True,
        fused=True,
    )


def layer(
    x,
    filter_output,
    filter_size=[1, 1],
    stride=[1, 1],
    pool=None,
    activate="bin",
    norm=True,
    epsilon=0.0001,
    padding="SAME",
):
    shape = filter_size + [x.shape[-1].value, filter_output]
    W, b = weight_bias(shape)
    x = tf.nn.conv2d(x, W, strides=[1, *stride, 1], padding=padding) + b
    if activate == "bin":
        if pool is not None:
            x = tf.nn.max_pool(
                x, ksize=[1, *pool[0], 1], strides=[1, *pool[-1], 1], padding="VALID"
            )
        if norm:
            x = batch_norm(x, epsilon)
    else:
        if norm:
            x = batch_norm(x, epsilon)
        if pool is not None:
            x = tf.nn.max_pool(
                x, ksize=[1, *pool[0], 1], strides=[1, *pool[-1], 1], padding="VALID"
            )
    if activate == "bin":
        return activation(x)
    else:
        return x


class Model:
    def __init__(self):
        self.LEARNING_RATE = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.Y = tf.placeholder(tf.float32, [None, 10])
        feed = layer(self.X, 32, filter_size=[3, 3])
        feed = layer(feed, 32, filter_size=[3, 3], pool=([2, 2], [2, 2]))
        feed = layer(feed, 64, filter_size=[3, 3])
        feed = layer(feed, 64, filter_size=[3, 3], pool=([2, 2], [2, 2]))
        feed = layer(feed, 128, filter_size=[3, 3])
        feed = layer(feed, 128, filter_size=[3, 3], pool=([2, 2], [2, 2]))
        feed = layer(feed, 512, filter_size=[3, 3], padding="VALID")
        feed = layer(feed, 512)
        self.logits = tf.reshape(layer(feed, 10, activate="none"), (-1, 10))
        self.cost = tf.reduce_mean(tf.square(tf.losses.hinge_loss(self.logits, self.Y)))
        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.logits, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[4]:


sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())


# In[5]:


EPOCH = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 128
LR_DECAY = (0.0000003 / LEARNING_RATE) ** (1.0 / EPOCH)


# In[6]:


for i in range(EPOCH):
    last = time.time()
    TOTAL_LOSS, ACCURACY = 0, 0
    for n in range(0, (mnist.train.images.shape[0] // BATCH_SIZE) * BATCH_SIZE, BATCH_SIZE):
        batch_x = mnist.train.images[n : n + BATCH_SIZE, :].reshape((-1, 28, 28, 1))
        acc, cost, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={
                model.X: batch_x,
                model.Y: mnist.train.labels[n : n + BATCH_SIZE, :],
                model.LEARNING_RATE: LEARNING_RATE,
            },
        )
        ACCURACY += acc
        TOTAL_LOSS += cost
    LEARNING_RATE *= LR_DECAY
    TOTAL_LOSS /= mnist.train.images.shape[0] // BATCH_SIZE
    ACCURACY /= mnist.train.images.shape[0] // BATCH_SIZE
    print(
        "epoch %d, avg loss %f, avg acc %f, time taken %f secs"
        % (i + 1, TOTAL_LOSS, ACCURACY, time.time() - last)
    )


# In[ ]:
