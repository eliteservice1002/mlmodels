#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import pickle
import re
import time

import numpy as np
import sklearn.datasets
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

# In[2]:


class Model:
    def __init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate):
        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(size_layer)

        self.rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

        self.X = tf.placeholder(tf.float32, [None, None, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])

        # dropout 0.5
        drop = tf.contrib.rnn.DropoutWrapper(self.rnn_cells, output_keep_prob=0.5)

        self.outputs, self.last_state = tf.nn.dynamic_rnn(drop, self.X, dtype=tf.float32)

        self.rnn_W = tf.Variable(tf.random_normal((size_layer, dimension_output)))
        self.rnn_B = tf.Variable(tf.random_normal([dimension_output]))
        self.logits = tf.matmul(self.outputs[:, -1], self.rnn_W) + self.rnn_B

        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )

        # L2 normalized
        l2 = sum(0.0005 * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

        self.cost += l2

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


# In[3]:


def clearstring(string):
    string = re.sub("[^\"'A-Za-z0-9 ]+", "", string)
    string = string.split(" ")
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = " ".join(string)
    return string


# because of sklean.datasets read a document as a single element
# so we want to split based on new line
def separate_dataset(trainset):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split("\n")
        # python3, if python2, just remove list()
        data_ = list(filter(None, data_))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget


# In[4]:


trainset_data = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset_data.data, trainset_data.target = separate_dataset(trainset_data)


# In[5]:


with open("dictionary_emotion.p", "rb") as fopen:
    dict_emotion = pickle.load(fopen)


# In[6]:


len_sentences = np.array([len(i.split()) for i in trainset_data.data])
maxlen = np.ceil(len_sentences.mean()).astype("int")
data_X = np.zeros((len(trainset_data.data), maxlen))


# In[7]:


for i in range(data_X.shape[0]):
    tokens = trainset_data.data[i].split()[:maxlen]
    for no, text in enumerate(tokens[::-1]):
        try:
            data_X[i, -1 - no] = dict_emotion[text]
        except:
            continue


# In[8]:


train_X, test_X, train_Y, test_Y = train_test_split(data_X, trainset_data.target, test_size=0.2)


# In[9]:


tf.reset_default_graph()
model = Model(3, 128, 1, len(trainset_data.target_names), 0.0001)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 100, 0, 0, 0
batch_size = 120
while True:
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:", EPOCH)
        break
    train_acc, train_loss = 0, 0
    for n in range(0, (train_X.shape[0] // batch_size) * batch_size, batch_size):
        batch_x = np.expand_dims(train_X[n : n + batch_size, :], axis=2)
        batch_y = np.zeros((batch_size, len(trainset_data.target_names)))
        for k in range(batch_size):
            batch_y[k, train_Y[n + k]] = 1.0
        _, loss = sess.run(
            [model.optimizer, model.cost], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        train_acc += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})
        train_loss += loss
    batch_y = np.zeros((test_X.shape[0], len(trainset_data.target_names)))
    for k in range(test_X.shape[0]):
        batch_y[k, test_Y[k]] = 1.0
    TEST_COST = sess.run(
        model.cost, feed_dict={model.X: np.expand_dims(test_X, axis=2), model.Y: batch_y}
    )
    TEST_ACC = sess.run(
        model.accuracy, feed_dict={model.X: np.expand_dims(test_X, axis=2), model.Y: batch_y}
    )
    train_loss /= train_X.shape[0] // batch_size
    train_acc /= train_X.shape[0] // batch_size
    if TEST_ACC > CURRENT_ACC:
        print("epoch:", EPOCH, ", pass acc:", CURRENT_ACC, ", current acc:", TEST_ACC)
        CURRENT_ACC = TEST_ACC
        saver.save(sess, os.getcwd() + "/model-rnn.ckpt")
    else:
        CURRENT_CHECKPOINT += 1
    EPOCH += 1
    print("epoch:", EPOCH, ", training loss: ", train_loss, ", train acc: ", train_acc)


# In[ ]:
