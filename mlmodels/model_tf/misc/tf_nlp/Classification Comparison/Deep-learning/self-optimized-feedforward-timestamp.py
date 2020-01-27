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

from bayes_opt import BayesianOptimization

# In[2]:


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


# In[3]:


trainset_data = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset_data.data, trainset_data.target = separate_dataset(trainset_data)


# In[4]:


with open("dictionary_emotion.p", "rb") as fopen:
    dict_emotion = pickle.load(fopen)


# In[5]:


# 30% of our dataset will be used for bayesian optimization
_, opt_X, _, opt_Y = train_test_split(trainset_data.data, trainset_data.target, test_size=0.1)
train_opt_X, test_opt_X, train_opt_Y, test_opt_Y = train_test_split(opt_X, opt_Y, test_size=0.2)


# In[6]:


class neuralnet:
    def __init__(self, timestamp, num_hidden, size_layer, learning_rate=0.01):
        def activate(first_layer, second_layer, bias):
            return tf.nn.relu(tf.matmul(first_layer, second_layer) + bias)

        self.X = tf.placeholder(tf.float32, (None, timestamp))
        self.Y = tf.placeholder(tf.float32, (None, len(trainset_data.target_names)))
        input_layer = tf.Variable(tf.random_normal([timestamp, size_layer]))
        biased_layer = tf.Variable(tf.random_normal([size_layer], stddev=0.1))
        output_layer = tf.Variable(tf.random_normal([size_layer, len(trainset_data.target_names)]))
        biased_output = tf.Variable(tf.random_normal([len(trainset_data.target_names)], stddev=0.1))
        layers, biased = [], []
        for i in range(num_hidden - 1):
            layers.append(tf.Variable(tf.random_normal([size_layer, size_layer])))
            biased.append(tf.Variable(tf.random_normal([size_layer])))
        first_l = activate(self.X, input_layer, biased_layer)
        next_l = activate(first_l, layers[0], biased[0])
        for i in range(1, num_hidden - 1):
            next_l = activate(next_l, layers[i], biased[i])
        self.last_l = tf.matmul(next_l, output_layer) + biased_output
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.last_l, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.last_l, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[7]:


def neural_network(
    timestamp, num_hidden, size_layer, learning_rate=0.001, batch_size=200, epoch=20
):
    tf.reset_default_graph()
    model = neuralnet(timestamp, num_hidden, size_layer, learning_rate)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    COST, TEST_COST, ACC, TEST_ACC = [], [], [], []
    for i in range(epoch):
        train_acc, train_loss = 0, 0
        for n in range(0, (len(train_opt_X) // batch_size) * batch_size, batch_size):
            batch_x = np.zeros((batch_size, timestamp))
            batch_y = np.zeros((batch_size, len(trainset_data.target_names)))
            for k in range(batch_size):
                tokens = train_opt_X[n + k].split()[:timestamp]
                for no, text in enumerate(tokens[::-1]):
                    try:
                        batch_x[k, -1 - no] = dict_emotion[text]
                    except:
                        continue
                batch_y[k, train_opt_Y[n + k]] = 1.0
            batch_x = StandardScaler().fit_transform(batch_x.T).T
            _, loss = sess.run(
                [model.optimizer, model.cost], feed_dict={model.X: batch_x, model.Y: batch_y}
            )
            train_acc += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})
            train_loss += loss
        batch_x = np.zeros((len(test_opt_X), timestamp))
        batch_y = np.zeros((len(test_opt_X), len(trainset_data.target_names)))
        for k in range(len(test_opt_X)):
            tokens = test_opt_X[k].split()[:timestamp]
            for no, text in enumerate(tokens[::-1]):
                try:
                    batch_x[k, -1 - no] = dict_emotion[text]
                except:
                    continue
            batch_y[k, test_opt_Y[k]] = 1.0
        batch_x = StandardScaler().fit_transform(batch_x.T).T
        TEST_COST.append(sess.run(model.cost, feed_dict={model.X: batch_x, model.Y: batch_y}))
        TEST_ACC.append(sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y}))
        train_loss /= len(train_opt_X) // batch_size
        train_acc /= len(train_opt_X) // batch_size
        ACC.append(train_acc)
        COST.append(train_loss)
    COST = np.array(COST).mean()
    TEST_COST = np.array(TEST_COST).mean()
    ACC = np.array(ACC).mean()
    TEST_ACC = np.array(TEST_ACC).mean()
    return COST, TEST_COST, ACC, TEST_ACC


# In[8]:


def generate_nn(timestamp, num_hidden, size_layer):
    global accbest
    param = {
        "timestamp": int(np.around(timestamp)),
        "num_hidden": int(np.around(num_hidden)),
        "size_layer": int(np.around(size_layer)),
    }
    print("\nSearch parameters %s" % (param), file=log_file)
    log_file.flush()
    learning_cost, valid_cost, learning_acc, valid_acc = neural_network(**param)
    print(
        "stop after 100 iteration with train cost %f, valid cost %f, train acc %f, valid acc %f"
        % (learning_cost, valid_cost, learning_acc, valid_acc)
    )
    if valid_acc > accbest:
        costbest = valid_acc
    return valid_acc


# In[9]:


log_file = open("nn-bayesian.log", "a")
accbest = 0.0
NN_BAYESIAN = BayesianOptimization(
    generate_nn, {"timestamp": (5, 50), "num_hidden": (2, 20), "size_layer": (32, 1024)}
)
NN_BAYESIAN.maximize(init_points=10, n_iter=20, acq="ei", xi=0.0)


# stop after 100 iteration with train cost 5.624642, valid cost 4.227495, train acc 0.317768, valid acc 0.319677
#    11 | 00m25s |    0.31968 |       2.0000 |      32.0000 |     50.0000 |

# In[11]:


data_X = np.zeros((len(trainset_data.data), 50))
for i in range(data_X.shape[0]):
    tokens = trainset_data.data[i].split()[:50]
    for no, text in enumerate(tokens[::-1]):
        try:
            data_X[i, -1 - no] = dict_emotion[text]
        except:
            continue
train_X, test_X, train_Y, test_Y = train_test_split(data_X, trainset_data.target, test_size=0.2)


# In[17]:


tf.reset_default_graph()
model = neuralnet(50, 2, 32)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 100, 0, 0, 0
batch_size = 200
while True:
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:", EPOCH)
        break
    train_acc, train_loss = 0, 0
    for n in range(0, (train_X.shape[0] // batch_size) * batch_size, batch_size):
        batch_x = train_X[n : n + batch_size, :]
        batch_y = np.zeros((batch_size, len(trainset_data.target_names)))
        for k in range(batch_size):
            batch_y[k, train_Y[n + k]] = 1.0
        batch_x = StandardScaler().fit_transform(batch_x.T).T
        _, loss = sess.run(
            [model.optimizer, model.cost], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        train_acc += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})
        train_loss += loss
    batch_y = np.zeros((test_X.shape[0], len(trainset_data.target_names)))
    for k in range(test_X.shape[0]):
        batch_y[k, test_Y[k]] = 1.0
    batch_x = StandardScaler().fit_transform(test_X.T).T
    TEST_COST = sess.run(model.cost, feed_dict={model.X: batch_x, model.Y: batch_y})
    TEST_ACC = sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})
    train_loss /= train_X.shape[0] // batch_size
    train_acc /= train_X.shape[0] // batch_size
    if TEST_ACC > CURRENT_ACC:
        print("epoch:", EPOCH, ", pass acc:", CURRENT_ACC, ", current acc:", TEST_ACC)
        CURRENT_ACC = TEST_ACC
        saver.save(sess, os.getcwd() + "/model.ckpt")
    else:
        CURRENT_CHECKPOINT += 1
    EPOCH += 1


# In[ ]:
