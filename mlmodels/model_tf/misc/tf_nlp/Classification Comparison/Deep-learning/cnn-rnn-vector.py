#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import os
import random
import re
import time

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

# In[3]:


maxlen = 20
location = os.getcwd()
num_layers = 3
size_layer = 256
learning_rate = 0.0001
batch = 100


# In[6]:


with open("dataset-emotion.p", "rb") as fopen:
    df = pickle.load(fopen)
with open("vector-emotion.p", "rb") as fopen:
    vectors = pickle.load(fopen)
with open("dataset-dictionary.p", "rb") as fopen:
    dictionary = pickle.load(fopen)
label = np.unique(df[:, 1])


# In[8]:


train_X, test_X, train_Y, test_Y = train_test_split(df[:, 0], df[:, 1], test_size=0.2)


# In[ ]:


class Model:
    def __init__(
        self,
        sequence_length,
        dimension_input,
        dimension_output,
        learning_rate,
        filter_sizes,
        pooling_size,
        out_dimension,
        num_layer,
    ):
        self.X = tf.placeholder(tf.float32, shape=[None, sequence_length, dimension_input, 1])
        self.Y = tf.placeholder(tf.float32, shape=[None, dimension_output])
        pooled_outputs = []
        reduce_size = int(np.ceil((sequence_length) * 1.0 / pooling_size))
        for i in filter_sizes:
            w = tf.Variable(tf.truncated_normal([i, dimension_input, 1, out_dimension], stddev=0.1))
            b = tf.Variable(tf.truncated_normal([out_dimension], stddev=0.01))
            conv = tf.nn.relu(tf.nn.conv2d(self.X, w, strides=[1, 1, 1, 1], padding="VALID") + b)
            pooled = tf.nn.max_pool(
                conv,
                ksize=[1, pooling_size, 1, 1],
                strides=[1, pooling_size, 1, 1],
                padding="VALID",
            )
            pooled = tf.reshape(pooled, [-1, reduce_size - 1, out_dimension])
            pooled_outputs.append(pooled)
        h_pool = tf.concat(pooled_outputs, 2)

        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(out_dimension)

        self.rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
        drop = tf.contrib.rnn.DropoutWrapper(self.rnn_cells, output_keep_prob=0.5)
        self.outputs, self.last_state = tf.nn.dynamic_rnn(drop, h_pool, dtype=tf.float32)
        self.rnn_W = tf.Variable(tf.random_normal((out_dimension, dimension_output)))
        self.rnn_B = tf.Variable(tf.random_normal([dimension_output]))
        self.logits = tf.matmul(self.outputs[:, -1], self.rnn_W) + self.rnn_B
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        l2 = sum(0.0005 * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        self.cost += l2
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


# In[16]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
dimension = vectors.shape[1]
model = Model(maxlen, dimension, len(label), learning_rate, [3, 3, 3], 5, size_layer, num_layers)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 10, 0, 0, 0
while True:
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:", EPOCH)
        break
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (train_X.shape[0] // batch) * batch, batch):
        batch_x = np.zeros((batch, maxlen, dimension))
        batch_y = np.zeros((batch, len(label)))
        for k in range(batch):
            tokens = train_X[i + k].split()[:maxlen]
            for no, text in enumerate(tokens[::-1]):
                try:
                    batch_x[k, -1 - no, :] += vectors[dictionary[text], :]
                except Exception as e:
                    print(e)
                    continue
            batch_y[k, int(train_Y[i + k])] = 1.0
        batch_x = np.expand_dims(batch_x, axis=-1)
        loss, _ = sess.run(
            [model.cost, model.optimizer], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        train_loss += loss
        train_acc += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})

    for i in range(0, (test_X.shape[0] // batch) * batch, batch):
        batch_x = np.zeros((batch, maxlen, dimension))
        batch_y = np.zeros((batch, len(label)))
        for k in range(batch):
            tokens = test_X[i + k].split()[:maxlen]
            for no, text in enumerate(tokens[::-1]):
                try:
                    batch_x[k, -1 - no, :] += vectors[dictionary[text], :]
                except:
                    continue
            batch_y[k, int(test_Y[i + k])] = 1.0
        batch_x = np.expand_dims(batch_x, axis=-1)
        loss, acc = sess.run(
            [model.cost, model.accuracy], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        test_loss += loss
        test_acc += acc

    train_loss /= train_X.shape[0] // batch
    train_acc /= train_X.shape[0] // batch
    test_loss /= test_X.shape[0] // batch
    test_acc /= test_X.shape[0] // batch
    if test_acc > CURRENT_ACC:
        print("epoch:", EPOCH, ", pass acc:", CURRENT_ACC, ", current acc:", test_acc)
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
        saver.save(sess, os.getcwd() + "/model-cnn-vector.ckpt")
    else:
        CURRENT_CHECKPOINT += 1
    EPOCH += 1
    print(
        "epoch:",
        EPOCH,
        ", training loss:",
        train_loss,
        ", training acc:",
        train_acc,
        ", valid loss:",
        test_loss,
        ", valid acc:",
        test_acc,
    )


# In[ ]:
