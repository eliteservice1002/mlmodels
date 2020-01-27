#!/usr/bin/env python
# coding: utf-8

# In[5]:


import functools

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# In[3]:


mnist = input_data.read_data_sets("", validation_size=0)


# In[14]:


def differentiable_clip(inputs, alpha, rmin, rmax):
    return tf.sigmoid(-alpha * (inputs - rmin)) + tf.sigmoid(alpha * (inputs - rmax))


def double_thresholding(inputs, per_pixel=True):
    input_shape = inputs.shape.as_list()
    if per_pixel:
        r = tf.Variable(tf.random_normal(input_shape[1:], stddev=np.sqrt(1 / input_shape[-1])))
    axis = (1, 2) if len(input_shape) == 4 else (1,)
    rmin = tf.reduce_min(inputs, axis=axis, keep_dims=True) * r
    rmax = tf.reduce_max(inputs, axis=axis, keep_dims=True) * r
    alpha = 0.2
    return 0.5 + (inputs - 0.5) * differentiable_clip(inputs, alpha, rmin, rmax)


def conv(inputs, filters, kernel_size):
    w = tf.Variable(
        tf.random_normal(
            [kernel_size, kernel_size, int(inputs.shape[-1]), filters], stddev=np.sqrt(1 / filters)
        )
    )
    conv = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="VALID")
    l = tf.constant(
        functools.reduce(lambda x, y: x * y, w.shape.as_list()[:3], 1), dtype=tf.float32
    )
    mean_weight = tf.constant(
        1, shape=[kernel_size, kernel_size, inputs.shape.as_list()[-1], 1], dtype=tf.float32
    )
    mean_x = 1.0 / l * tf.nn.conv2d(inputs, mean_weight, strides=[1, 1, 1, 1], padding="VALID")
    mean_w = tf.reduce_mean(w, axis=(0, 1, 2), keep_dims=True)
    hout = (2.0 / l) * conv - mean_w - mean_x
    return double_thresholding(hout)


def fully_connected(inputs, out_size):
    w = tf.Variable(
        tf.random_normal([int(inputs.shape[-1]), out_size], stddev=np.sqrt(1 / out_size))
    )
    l = tf.constant(inputs.shape.as_list()[1], dtype=tf.float32)
    mean_x = tf.reduce_mean(inputs, axis=1, keep_dims=True)
    mean_w = tf.reduce_mean(w, axis=0, keep_dims=True)
    hout = (2.0 / l) * tf.matmul(inputs, w) - mean_w - mean_x
    return double_thresholding(hout)


class Model:
    def __init__(self, learning_rate):
        self.X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self.Y = tf.placeholder(tf.float32, shape=[None, 10])
        conv1 = tf.nn.relu(conv(self.X, 16, 5))
        pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])
        conv2 = tf.nn.relu(conv(pool1, 64, 5))
        pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2])
        pool2_shape = pool2.shape.as_list()
        pulled_pool2 = tf.reshape(pool2, [-1, pool2_shape[1] * pool2_shape[2] * pool2_shape[3]])
        fc1 = tf.nn.relu(fully_connected(pulled_pool2, 1024))
        self.logits = fully_connected(fc1, 10)
        self.cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.logits)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[21]:


batch_size = 128
learning_rate = 0.2
epoch = 10

train_images = mnist.train.images.reshape((-1, 28, 28, 1))
test_images = mnist.test.images.reshape((-1, 28, 28, 1))

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(learning_rate)
sess.run(tf.global_variables_initializer())


# In[22]:


LOSS, ACC_TRAIN, ACC_TEST = [], [], []
for i in range(epoch):
    total_loss, total_acc = 0, 0
    for n in range(0, (mnist.train.images.shape[0] // batch_size) * batch_size, batch_size):
        batch_x = train_images[n : n + batch_size, :, :, :]
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
        batch_x = test_images[n : n + batch_size, :, :, :]
        batch_y = np.zeros((batch_size, 10))
        batch_y[np.arange(batch_size), mnist.test.labels[n : n + batch_size]] = 1.0
        total_acc += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})
    total_acc /= mnist.test.images[:1000, :].shape[0] // batch_size
    ACC_TEST.append(total_acc)
    print(
        "epoch: %d, accuracy train: %f, accuracy testing: %f" % (i + 1, ACC_TRAIN[-1], ACC_TEST[-1])
    )


# In[ ]:
