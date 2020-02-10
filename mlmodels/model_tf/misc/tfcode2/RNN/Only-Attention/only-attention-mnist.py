#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# In[2]:


mnist = input_data.read_data_sets("", one_hot=True)


# In[4]:


def sinusoidal_positional_encoding(inputs, num_units, zero_pad=False, scale=False):
    T = inputs.get_shape().as_list()[1]
    position_idx = tf.tile(tf.expand_dims(tf.range(T), 0), [tf.shape(inputs)[0], 1])
    position_enc = np.array(
        [[pos / np.power(10000, 2.0 * i / num_units) for i in range(num_units)] for pos in range(T)]
    )
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
    lookup_table = tf.convert_to_tensor(position_enc, tf.float32)
    if zero_pad:
        lookup_table = tf.concat([tf.zeros([1, num_units]), lookup_table[1:, :]], axis=0)
    outputs = tf.nn.embedding_lookup(lookup_table, position_idx)
    if scale:
        outputs = outputs * num_units ** 0.5
    return outputs


class Model:
    def __init__(self):
        dimension_input = 28
        dimension_output = 10
        self.X = tf.placeholder(tf.float32, [None, dimension_input, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        x = self.X
        x += sinusoidal_positional_encoding(x, dimension_input)
        masks = tf.sign(self.X[:, :, 0])
        align = tf.squeeze(tf.layers.dense(x, 1, tf.tanh), -1)
        paddings = tf.fill(tf.shape(align), float("-inf"))
        align = tf.where(tf.equal(masks, 0), paddings, align)
        align = tf.expand_dims(tf.nn.softmax(align), -1)
        x = tf.squeeze(tf.matmul(tf.transpose(x, [0, 2, 1]), align), -1)
        self.logits = tf.layers.dense(x, dimension_output)
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


# In[5]:


sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())


# In[6]:


EPOCH = 10
BATCH_SIZE = 128


# In[7]:


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
