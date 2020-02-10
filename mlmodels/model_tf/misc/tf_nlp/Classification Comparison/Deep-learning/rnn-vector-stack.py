#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import os
import pickle
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
learning_rate = 0.00001
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


class Model:
    def __init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate):
        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(size_layer)

        self.X = tf.placeholder(tf.float32, [None, None, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])

        with tf.variable_scope("hinge", reuse=False):
            rnn_cells_hinge = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
            drop_hinge = tf.contrib.rnn.DropoutWrapper(rnn_cells_hinge, output_keep_prob=0.5)
            self.outputs_hinge, _ = tf.nn.dynamic_rnn(drop_hinge, self.X, dtype=tf.float32)
            rnn_W_hinge = tf.Variable(tf.random_normal((size_layer, dimension_output)))
            rnn_B_hinge = tf.Variable(tf.random_normal([dimension_output]))
            self.logits_hinge = tf.matmul(self.outputs_hinge[:, -1], rnn_W_hinge) + rnn_B_hinge

        with tf.variable_scope("huber", reuse=False):
            rnn_cells_huber = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
            drop_huber = tf.contrib.rnn.DropoutWrapper(rnn_cells_huber, output_keep_prob=0.5)
            self.outputs_huber, _ = tf.nn.dynamic_rnn(drop_huber, self.X, dtype=tf.float32)
            rnn_W_huber = tf.Variable(tf.random_normal((size_layer, dimension_output)))
            rnn_B_huber = tf.Variable(tf.random_normal([dimension_output]))
            self.logits_huber = tf.matmul(self.outputs_huber[:, -1], rnn_W_huber) + rnn_B_huber

        rnn_W = tf.Variable(tf.random_normal((dimension_output * 2, dimension_output)))
        rnn_B = tf.Variable(tf.random_normal([dimension_output]))
        self.logits = tf.matmul(tf.concat([self.logits_hinge, self.logits_huber], 1), rnn_W) + rnn_B

        cost_hinge = tf.losses.hinge_loss(logits=self.logits_hinge, labels=self.Y)
        cost_huber = tf.losses.huber_loss(predictions=self.logits_huber, labels=self.Y)
        cost_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.cost = 0.5 * cost_entropy + 0.25 * cost_huber + 0.25 * cost_hinge
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


# In[ ]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(num_layers, size_layer, vectors.shape[1], label.shape[0], learning_rate)
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
        saver.save(sess, os.getcwd() + "/model-rnn-vector-huber.ckpt")
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
