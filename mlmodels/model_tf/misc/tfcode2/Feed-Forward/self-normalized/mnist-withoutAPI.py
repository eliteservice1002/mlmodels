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


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def alpha_dropout(x, rate, alpha=-1.7580993408473766, mean=0.0, var=1.0):
    keep_prob = 1.0 - rate
    random_tensor = tf.random_uniform(tf.shape(x)) + keep_prob
    binary_tensor = tf.floor(random_tensor)
    ret = x * binary_tensor + alpha * (1 - binary_tensor)
    a = tf.sqrt(var / (keep_prob * ((1 - keep_prob) * tf.pow(alpha - mean, 2) + var)))
    b = mean - a * (keep_prob * mean + (1 - keep_prob) * alpha)
    return a * ret + b


# In[4]:


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
        feedforward = selu(tf.matmul(self.X, w1) + b1)
        feeddropout = alpha_dropout(feedforward, 0.5)
        feedforward = selu(tf.matmul(feeddropout, w2) + b2)
        feeddropout = alpha_dropout(feedforward, 0.5)
        self.logits = tf.matmul(feeddropout, w3) + b3
        self.cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.logits)
        )
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[5]:


batch_size = 128
epoch = 10

train_images = (mnist.train.images - np.mean(mnist.train.images)) / np.std(mnist.train.images)
test_images = (mnist.test.images - np.mean(mnist.test.images)) / np.std(mnist.test.images)

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())


# In[6]:


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
