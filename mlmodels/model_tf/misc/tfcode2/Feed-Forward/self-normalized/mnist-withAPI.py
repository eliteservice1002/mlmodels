#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

sns.set()


# In[2]:


mnist = input_data.read_data_sets("", validation_size=0)


# In[3]:


class Model:
    def __init__(self):
        self.X = tf.placeholder(tf.float32, [None, 784])
        self.Y = tf.placeholder(tf.float32, [None, 10])
        w1 = tf.Variable(tf.random_normal([784, 256], stddev=np.sqrt(1 / 784)))
        b1 = tf.Variable(tf.random_normal([256], stddev=0))
        w2 = tf.Variable(tf.random_normal([256, 100], stddev=np.sqrt(1 / 256)))
        b2 = tf.Variable(tf.random_normal([100], stddev=0))
        w3 = tf.Variable(tf.random_normal([100, 10], stddev=np.sqrt(1 / 100)))
        b3 = tf.Variable(tf.random_normal([10], stddev=0))
        feedforward = tf.nn.selu(tf.matmul(self.X, w1) + b1)
        feeddropout = tf.contrib.nn.alpha_dropout(feedforward, 0.5)
        feedforward = tf.nn.selu(tf.matmul(feeddropout, w2) + b2)
        feeddropout = tf.contrib.nn.alpha_dropout(feedforward, 0.5)
        self.logits = tf.matmul(feeddropout, w3) + b3
        self.cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.logits)
        )
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[4]:


batch_size = 128
epoch = 10

train_images = (mnist.train.images - np.mean(mnist.train.images)) / np.std(mnist.train.images)
test_images = (mnist.test.images - np.mean(mnist.test.images)) / np.std(mnist.test.images)

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())


# In[5]:


LOSS, ACC_TRAIN, ACC_TEST = [], [], []
for i in range(epoch):
    total_loss, total_acc = 0, 0
    for n in range(0, (mnist.train.images.shape[0] // batch_size) * batch_size, batch_size):
        batch_x = train_images[n : n + batch_size, :]
        batch_y = np.zeros((batch_size, 10))
        batch_y[np.arange(batch_size), mnist.train.labels[n : n + batch_size]] = 1.0
        cost, _ = sess.run(
            [model.cost, model.optimizer], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        total_acc += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})
        total_loss += cost
    total_loss /= mnist.train.images.shape[0] // batch_size
    total_acc /= mnist.train.images.shape[0] // batch_size
    ACC_TRAIN.append(total_acc)
    total_acc = 0
    for n in range(
        0, (mnist.test.images[:1000, :].shape[0] // batch_size) * batch_size, batch_size
    ):
        batch_x = test_images[n : n + batch_size, :]
        batch_y = np.zeros((batch_size, 10))
        batch_y[np.arange(batch_size), mnist.test.labels[n : n + batch_size]] = 1.0
        total_acc += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})
    total_acc /= mnist.test.images[:1000, :].shape[0] // batch_size
    ACC_TEST.append(total_acc)
    print(
        "epoch: %d, accuracy train: %f, accuracy testing: %f" % (i + 1, ACC_TRAIN[-1], ACC_TEST[-1])
    )


# In[ ]:
