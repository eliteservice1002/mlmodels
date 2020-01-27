#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.examples.tutorials.mnist import input_data

from tflearn.layers.conv import global_avg_pool

# In[2]:


mnist = input_data.read_data_sets("", one_hot=True)


# In[3]:


# In[4]:


growth_k = 12
nb_block = 2
learning_rate = 1e-4
epsilon = 1e-8
dropout_rate = 0.2
class_num = 10
batch_size = 128
epoch = 50


# In[ ]:


sess = tf.InteractiveSession()


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(
            inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding="SAME"
        )
        return network


def global_average_pooling(x, stride=1):
    return global_avg_pool(x, name="gop")


def batch_normalization(x, training, scope):
    with arg_scope(
        [batch_norm],
        scope=scope,
        updates_collections=None,
        decay=0.9,
        center=True,
        scale=True,
        zero_debias_moving_mean=True,
    ):
        return tf.cond(
            training,
            lambda: batch_norm(inputs=x, is_training=training, reuse=None),
            lambda: batch_norm(inputs=x, is_training=training, reuse=True),
        )


def drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def average_pooling(x, pool_size=[2, 2], stride=2, padding="VALID"):
    return tf.layers.average_pooling2d(
        inputs=x, pool_size=pool_size, strides=stride, padding=padding
    )


def max_pooling(x, pool_size=[3, 3], stride=2, padding="VALID"):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def concatenation(layers):
    return tf.concat(layers, axis=3)


def linear(x):
    return tf.layers.dense(inputs=x, units=class_num, name="linear")


class DenseNet:
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization(x, training=self.training, scope=scope + "_batch1")
            x = tf.nn.relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope + "_conv1")
            x = drop_out(x, rate=dropout_rate, training=self.training)
            x = batch_normalization(x, training=self.training, scope=scope + "_batch2")
            x = tf.nn.relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope + "_conv2")
            x = drop_out(x, rate=dropout_rate, training=self.training)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization(x, training=self.training, scope=scope + "_batch1")
            x = tf.nn.relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1, 1], layer_name=scope + "_conv1")
            x = drop_out(x, rate=dropout_rate, training=self.training)
            x = average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + "_bottleN_" + str(0))
            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + "_bottleN_" + str(i + 1))
                layers_concat.append(x)

            x = concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(
            input_x, filter=2 * self.filters, kernel=[7, 7], stride=2, layer_name="conv0"
        )
        x = max_pooling(x, pool_size=[3, 3], stride=2)

        for i in range(self.nb_blocks):
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name="dense_" + str(i))
            x = self.transition_layer(x, scope="trans_" + str(i))

        x = self.dense_block(input_x=x, nb_layers=32, layer_name="dense_final")
        x = batch_normalization(x, training=self.training, scope="linear_batch")
        x = tf.nn.relu(x)
        x = global_average_pooling(x)
        x = flatten(x)
        x = linear(x)
        return x


x = tf.placeholder(tf.float32, shape=[None, 784])
batch_images = tf.reshape(x, [-1, 28, 28, 1])
label = tf.placeholder(tf.float32, shape=[None, 10])
training_flag = tf.placeholder(tf.bool)
logits = DenseNet(
    x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag
).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon).minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


# In[ ]:


LOSS, ACC_TRAIN, ACC_TEST = [], [], []
for i in range(epoch):
    total_loss, total_acc = 0, 0
    for n in range(0, (mnist.train.images.shape[0] // batch_size) * batch_size, batch_size):
        batch_x = mnist.train.images[n : n + batch_size]
        batch_y = mnist.train.labels[n : n + batch_size]
        loss, _ = sess.run(
            [cost, optimizer], feed_dict={x: batch_x, label: batch_y, training_flag: True}
        )
        total_acc += sess.run(
            accuracy, feed_dict={x: batch_x, label: batch_y, training_flag: False}
        )
        total_loss += loss
    total_loss /= mnist.train.images.shape[0] // batch_size
    total_acc /= mnist.train.images.shape[0] // batch_size
    ACC_TRAIN.append(total_acc)
    total_acc = 0
    for n in range(
        0, (mnist.test.images[:1000, :].shape[0] // batch_size) * batch_size, batch_size
    ):
        batch_x = mnist.test.images[n : n + batch_size]
        batch_y = mnist.test.labels[n : n + batch_size]
        total_acc += sess.run(
            accuracy, feed_dict={x: batch_x, label: batch_y, training_flag: False}
        )
    total_acc /= mnist.test.images[:1000, :].shape[0] // batch_size
    ACC_TEST.append(total_acc)
    print(
        "epoch: %d, accuracy train: %f, accuracy testing: %f" % (i + 1, ACC_TRAIN[-1], ACC_TEST[-1])
    )
