#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug

# In[2]:


epoch = 10
batch_size = 32
mnist = input_data.read_data_sets("", one_hot=True)
sess = tf.InteractiveSession()


def feed_dict(train):
    x, y = mnist.train.next_batch(batch_size)
    return {X: x, Y: y}


def convolutionize(x, conv_w, h=1):
    return tf.nn.conv2d(input=x, filter=conv_w, strides=[1, h, h, 1], padding="SAME")


def pooling(wx):
    return tf.nn.max_pool(wx, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


with tf.name_scope("input"):
    X = tf.placeholder(tf.float32, [None, 28, 28, 1], name="x-input")
    Y = tf.placeholder(tf.float32, [None, 10], name="y-input")

with tf.name_scope("conv_1"):
    with tf.name_scope("weights"):
        w1 = tf.Variable(tf.random_normal([3, 3, 1, 16], stddev=0.5))
    with tf.name_scope("biases"):
        b1 = tf.Variable(tf.zeros(shape=[16]))
    with tf.name_scope("activate"):
        conv1 = pooling(tf.nn.sigmoid(convolutionize(X, w1) + b1))

with tf.name_scope("conv_2"):
    with tf.name_scope("weights"):
        w2 = tf.Variable(tf.random_normal([3, 3, 16, 8], stddev=0.5))
    with tf.name_scope("biases"):
        b2 = tf.Variable(tf.zeros(shape=[8]))
    with tf.name_scope("activate"):
        conv2 = pooling(tf.nn.sigmoid(convolutionize(conv1, w2) + b2))

with tf.name_scope("conv_3"):
    with tf.name_scope("weights"):
        w3 = tf.Variable(tf.random_normal([3, 3, 8, 8], stddev=0.5))
    with tf.name_scope("biases"):
        b3 = tf.Variable(tf.zeros(shape=[8]))
    with tf.name_scope("activate"):
        conv3 = pooling(tf.nn.sigmoid(convolutionize(conv2, w3) + b3))
        conv3 = tf.reshape(conv3, [-1, 128])

with tf.name_scope("logits"):
    with tf.name_scope("weights"):
        w4 = tf.Variable(tf.random_normal([128, 10], stddev=0.5))
    with tf.name_scope("biases"):
        b4 = tf.Variable(tf.zeros(shape=[10]))
    with tf.name_scope("activate"):
        logits = tf.matmul(conv3, w4) + b4

with tf.name_scope("cross_entropy"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)

with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(logits, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("./logs", sess.graph)


# Open your terminal and execute
# ```bash
# tensorboard --logdir=./logs --port 6006 --debugger_port 6064
# ```

# # Open first tensorboard before run blocks below!

# In[3]:


sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6064")


# In[4]:


for i in range(epoch):
    xs, ys = mnist.train.next_batch(batch_size)
    xs = xs.reshape((-1, 28, 28, 1))
    sess.run(optimizer, feed_dict={X: xs, Y: ys})


# In[ ]:
