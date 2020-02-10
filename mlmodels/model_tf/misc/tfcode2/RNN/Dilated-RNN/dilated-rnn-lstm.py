#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import copy
import os
import pickle
import random
import re
import time

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder


def contruct_cells(hidden_structs):
    cells = []
    for hidden_dims in hidden_structs:
        cells.append(tf.contrib.rnn.LSTMCell(hidden_dims))
    return cells


def rnn_reformat(x, input_dims, n_steps):
    x_ = tf.transpose(x, [1, 0, 2])
    x_ = tf.reshape(x_, [-1, input_dims])
    return tf.split(x_, n_steps, 0)


def dilated_rnn(cell, inputs, rate, scope="default"):
    n_steps = len(inputs)
    if not (n_steps % rate) == 0:
        zero_tensor = tf.zeros_like(inputs[0])
        dialated_n_steps = n_steps // rate + 1
        for i_pad in range(dialated_n_steps * rate - n_steps):
            inputs.append(zero_tensor)
    else:
        dialated_n_steps = n_steps // rate
    dilated_inputs = [
        tf.concat(inputs[i * rate : (i + 1) * rate], axis=0) for i in range(dialated_n_steps)
    ]
    dilated_outputs, _ = tf.contrib.rnn.static_rnn(
        cell, dilated_inputs, dtype=tf.float32, scope=scope
    )
    splitted_outputs = [tf.split(output, rate, axis=0) for output in dilated_outputs]
    unrolled_outputs = [output for sublist in splitted_outputs for output in sublist]
    return unrolled_outputs[:n_steps]


def multi_dilated_rnn(cells, inputs, dilations):
    x = copy.copy(inputs)
    for cell, dilation in zip(cells, dilations):
        x = dilated_rnn(cell, x, dilation, scope="multi_dilated_rnn_%d" % dilation)
    return x


class Model:
    def __init__(
        self,
        steps,
        dimension_input,
        dimension_output,
        learning_rate=0.0001,
        hidden_structs=[20],
        dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    ):
        hidden_structs = hidden_structs * len(dilations)
        self.X = tf.placeholder(tf.float32, [None, steps, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        x_reformat = rnn_reformat(self.X, dimension_input, steps)
        cells = contruct_cells(hidden_structs)
        layer_outputs = multi_dilated_rnn(cells, x_reformat, dilations)
        if dilations[0] == 1:
            weights = tf.Variable(tf.random_normal(shape=[hidden_structs[-1], dimension_output]))
            bias = tf.Variable(tf.random_normal(shape=[dimension_output]))
            self.logits = tf.matmul(layer_outputs[-1], weights) + bias
        else:
            weights = tf.Variable(
                tf.random_normal(shape=[hidden_structs[-1] * dilations[0], dimension_output])
            )
            bias = tf.Variable(tf.random_normal(shape=[dimension_output]))
            for idx, i in enumerate(range(-dilations[0], 0, 1)):
                if idx == 0:
                    hidden_outputs_ = layer_outputs[i]
                else:
                    hidden_outputs_ = tf.concat([hidden_outputs_, layer_outputs[i]], axis=1)
            self.logits = tf.matmul(hidden_outputs_, weights) + bias
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


# In[2]:


# In[3]:


maxlen = 50
location = os.getcwd()
batch = 100


# In[4]:


with open("dataset-emotion.p", "rb") as fopen:
    df = pickle.load(fopen)
with open("vector-emotion.p", "rb") as fopen:
    vectors = pickle.load(fopen)
with open("dataset-dictionary.p", "rb") as fopen:
    dictionary = pickle.load(fopen)


# In[5]:


label = np.unique(df[:, 1])


# In[ ]:


train_X, test_X, train_Y, test_Y = train_test_split(df[:, 0], df[:, 1].astype("int"), test_size=0.2)


# In[ ]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(maxlen, vectors.shape[1], label.shape[0])
sess.run(tf.global_variables_initializer())
dimension = vectors.shape[1]
saver = tf.train.Saver(tf.global_variables())
EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 10, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:", EPOCH)
        break
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (train_X.shape[0] // batch) * batch, batch):
        batch_x = np.zeros((batch, maxlen, dimension))
        batch_y = np.zeros((batch, len(label)))
        for k in range(batch):
            tokens = train_X[i + k].split()[:maxlen]
            emb_data = np.zeros((maxlen, dimension), dtype=np.float32)
            for no, text in enumerate(tokens[::-1]):
                try:
                    emb_data[-1 - no, :] += vectors[dictionary[text], :]
                except Exception as e:
                    print(e)
                    continue
            batch_y[k, int(train_Y[i + k])] = 1.0
            batch_x[k, :, :] = emb_data[:, :]
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
            emb_data = np.zeros((maxlen, dimension), dtype=np.float32)
            for no, text in enumerate(tokens[::-1]):
                try:
                    emb_data[-1 - no, :] += vectors[dictionary[text], :]
                except:
                    continue
            batch_y[k, int(test_Y[i + k])] = 1.0
            batch_x[k, :, :] = emb_data[:, :]
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
        saver.save(sess, os.getcwd() + "/model-dilated-rnn-vector.ckpt")
    else:
        CURRENT_CHECKPOINT += 1
    EPOCH += 1
    print("time taken:", time.time() - lasttime)
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
