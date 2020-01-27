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


def squash(X, epsilon=1e-9):
    vec_squared_norm = tf.reduce_sum(tf.square(X), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    return scalar_factor * X


def conv_layer(X, num_output, num_vector, kernel=None, stride=None):
    global batch_size
    capsules = tf.contrib.layers.conv2d(
        X, num_output * num_vector, kernel, stride, padding="VALID", activation_fn=tf.nn.relu
    )
    capsules = tf.reshape(capsules, (batch_size, -1, num_vector, 1))
    return squash(capsules)


def routing(X, b_IJ, routing_times=2):
    global batch_size
    w = tf.Variable(tf.truncated_normal([1, 1152, 10, 8, 16], stddev=1e-1))
    X = tf.tile(X, [1, 1, 10, 1, 1])
    w = tf.tile(w, [batch_size, 1, 1, 1, 1])
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
            v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
            u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
            b_IJ += u_produce_v
    return v_J


def fully_conn_layer(X, num_output):
    global batch_size
    X_ = tf.reshape(X, shape=(batch_size, -1, 1, X.shape[-2].value, 1))
    b_IJ = tf.constant(np.zeros([batch_size, 1152, num_output, 1, 1], dtype=np.float32))
    capsules = routing(X_, b_IJ, routing_times=2)
    capsules = tf.squeeze(capsules, axis=1)
    return capsules


class CapsuleNetwork:
    def __init__(
        self,
        batch_size,
        learning_rate,
        regularization_scale=0.392,
        epsilon=1e-8,
        m_plus=0.9,
        m_minus=0.1,
        lambda_val=0.5,
    ):
        self.X = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        self.Y = tf.placeholder(tf.float32, shape=(None, 10))
        conv1 = tf.contrib.layers.conv2d(
            self.X, num_outputs=256, kernel_size=9, stride=1, padding="VALID"
        )
        caps1 = conv_layer(conv1, 32, 8, 9, 2)
        caps2 = fully_conn_layer(caps1, 10)
        v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True) + epsilon)
        self.logits = tf.nn.softmax(v_length, dim=1)[:, :, 0, 0]
        masked_v = tf.multiply(tf.squeeze(caps2), tf.reshape(self.Y, (-1, 10, 1)))
        v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True) + epsilon)
        vector_j = tf.reshape(masked_v, shape=(batch_size, -1))
        fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
        decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)
        max_l = tf.square(tf.maximum(0.0, m_plus - v_length))
        max_r = tf.square(tf.maximum(0.0, v_length - m_minus))
        max_l = tf.reshape(max_l, shape=(batch_size, -1))
        max_r = tf.reshape(max_r, shape=(batch_size, -1))
        L_c = self.Y * max_l + lambda_val * (1 - self.Y) * max_r
        margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
        origin = tf.reshape(self.X, shape=(batch_size, -1))
        squared = tf.reduce_mean(tf.square(decoded - origin))
        self.cost = margin_loss + regularization_scale * squared
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[ ]:


batch_size = 128
learning_rate = 0.001
epoch = 5

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = CapsuleNetwork(batch_size, learning_rate)
sess.run(tf.global_variables_initializer())


# In[ ]:


LOSS, ACC_TRAIN, ACC_TEST = [], [], []
for i in range(epoch):
    total_loss, total_acc = 0, 0
    for n in range(0, (mnist.train.images.shape[0] // batch_size) * batch_size, batch_size):
        batch_x = mnist.train.images[n : n + batch_size, :].reshape((-1, 28, 28, 1))
        batch_y = np.zeros((batch_size, 10))
        for k in range(batch_size):
            batch_y[k, mnist.train.labels[n + k]] = 1.0
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
        batch_x = mnist.test.images[n : n + batch_size, :].reshape((-1, 28, 28, 1))
        batch_y = np.zeros((batch_size, 10))
        for k in range(batch_size):
            batch_y[k, mnist.test.labels[n + k]] = 1.0
        total_acc += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})
    total_acc /= mnist.test.images[:1000, :].shape[0] // batch_size
    ACC_TEST.append(total_acc)
    print(
        "epoch: %d, accuracy train: %f, accuracy testing: %f" % (i + 1, ACC_TRAIN[-1], ACC_TEST[-1])
    )
