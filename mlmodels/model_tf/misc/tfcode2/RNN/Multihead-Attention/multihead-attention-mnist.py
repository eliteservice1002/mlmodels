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


def embed_seq(inputs, vocab_size=None, embed_dim=None, zero_pad=False, scale=False):
    lookup_table = tf.get_variable("lookup_table", dtype=tf.float32, shape=[vocab_size, embed_dim])
    if zero_pad:
        lookup_table = tf.concat((tf.zeros([1, embed_dim]), lookup_table[1:, :]), axis=0)
    outputs = tf.nn.embedding_lookup(lookup_table, inputs)
    if scale:
        outputs = outputs * (embed_dim ** 0.5)
    return outputs


def learned_positional_encoding(inputs, embed_dim, zero_pad=False, scale=False):
    T = inputs.get_shape().as_list()[1]
    outputs = tf.range(T)
    outputs = tf.expand_dims(outputs, 0)
    outputs = tf.tile(outputs, [tf.shape(inputs)[0], 1])
    return embed_seq(outputs, T, embed_dim, zero_pad=zero_pad, scale=scale)


def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))
    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable("gamma", params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable("beta", params_shape, tf.float32, tf.zeros_initializer())
    return gamma * normalized + beta


def pointwise_feedforward(inputs, num_units=[None, None], activation=None):
    outputs = tf.layers.conv1d(inputs, num_units[0], kernel_size=1, activation=activation)
    outputs = tf.layers.conv1d(outputs, num_units[1], kernel_size=1, activation=None)
    outputs += inputs
    outputs = layer_norm(outputs)
    return outputs


class Model:
    def __init__(
        self,
        dimension_output=10,
        learning_rate=1e-4,
        seq_len=28,
        dimension_input=28,
        num_heads=28,
        attn_windows=range(1, 6),
    ):
        self.size_layer = dimension_input
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.X = tf.placeholder(tf.float32, [None, seq_len, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        feed = self.X
        for i, win_size in enumerate(attn_windows):
            with tf.variable_scope("attn_masked_window_%d" % win_size):
                feed = self.multihead_attn(feed, self.window_mask(win_size))
        feed += learned_positional_encoding(feed, dimension_input)
        with tf.variable_scope("multihead"):
            feed = self.multihead_attn(feed, None)
        with tf.variable_scope("pointwise"):
            feed = pointwise_feedforward(
                feed, num_units=[4 * dimension_input, dimension_input], activation=tf.nn.relu
            )
        self.logits = tf.layers.dense(feed, dimension_output)[:, -1]
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def multihead_attn(self, inputs, masks):
        T_q = T_k = inputs.get_shape().as_list()[1]
        Q_K_V = tf.layers.dense(inputs, 3 * self.size_layer, tf.nn.relu)
        Q, K, V = tf.split(Q_K_V, 3, -1)
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)
        align = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        align = align / np.sqrt(K_.get_shape().as_list()[-1])
        if masks is not None:
            paddings = tf.fill(tf.shape(align), float("-inf"))
            align = tf.where(tf.equal(masks, 0), paddings, align)
        align = tf.nn.softmax(align)
        outputs = tf.matmul(align, V_)
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
        outputs += inputs
        return layer_norm(outputs)

    def window_mask(self, h_w):
        masks = np.zeros([self.seq_len, self.seq_len])
        for i in range(self.seq_len):
            if i < h_w:
                masks[i, : i + h_w + 1] = 1.0
            elif i > self.seq_len - h_w - 1:
                masks[i, i - h_w :] = 1.0
            else:
                masks[i, i - h_w : i + h_w + 1] = 1.0
        masks = tf.convert_to_tensor(masks)
        return tf.tile(tf.expand_dims(masks, 0), [tf.shape(self.X)[0] * self.num_heads, 1, 1])


# In[4]:


sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())


# In[5]:


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


# In[ ]:
