#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

from bayes_opt import BayesianOptimization

# In[2]:


def clearstring(string):
    string = re.sub("[^A-Za-z0-9 ]+", "", string)
    string = string.split(" ")
    string = filter(None, string)
    string = [y.strip() for y in string]
    length = len(string)
    string = " ".join(string)
    return string.lower()


# In[3]:


with open("rt-polarity.neg", "r") as fopen:
    negatives = fopen.read().split("\n")
with open("rt-polarity.pos", "r") as fopen:
    positives = fopen.read().split("\n")


# In[4]:


negatives = negatives[:50]
positives = positives[:50]

for i in range(len(negatives)):
    negatives[i] = clearstring(negatives[i])
    positives[i] = clearstring(positives[i])


# In[5]:


vocab = []
for i in range(len(negatives)):
    vocab += negatives[i].split()
    vocab += positives[i].split()

vocab = sorted(vocab, key=vocab.count, reverse=True)
d1 = dict((k, v) for v, k in enumerate(reversed(vocab)))
vocab = ["PAD", "EOS"] + sorted(d1, key=d1.get, reverse=True)
print("vocab size:", len(vocab))
dictionary = dict(zip(vocab, [i for i in range(len(vocab))]))


# In[6]:


x_data = negatives + positives
y_data = [0] * len(negatives) + [1] * len(positives)
onehot = np.zeros((len(negatives) + len(positives), 2))
for i in range(onehot.shape[0]):
    onehot[i, y_data[i]] = 1.0

x_train, x_test, y_train, y_test, y_train_label, y_test_label = train_test_split(
    x_data, onehot, y_data, test_size=0.20
)


# ```text
# Activation function:
# 0- for sigmoid
# 1- for tanh
# 2- for relu
#
# Now the constants are:
# 1- batch size : 20
# 2- epoch: 50
# 3- gradient descent
# 4- softmax with cross entropy
# ```
#
# So you can change anything you want

# In[7]:


def neural_network(
    num_hidden, size_layer, learning_rate, dropout_rate, beta, activation, seq_len, batch_size=20
):
    tf.reset_default_graph()

    def lstm_cell(size_layer, activation):
        if activation == 0:
            activation = tf.nn.sigmoid
        elif activation == 1:
            activation = tf.nn.tanh
        else:
            activation = tf.nn.relu
        return tf.nn.rnn_cell.LSTMCell(size_layer, activation=activation)

    rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_cell(size_layer, activation) for _ in range(num_hidden)]
    )
    X = tf.placeholder(tf.float32, [None, None, 1])
    Y = tf.placeholder(tf.float32, [None, np.unique(y_train).shape[0]])
    drop = tf.contrib.rnn.DropoutWrapper(rnn_cells, output_keep_prob=dropout_rate)
    outputs, _ = tf.nn.dynamic_rnn(drop, X, dtype=tf.float32)
    rnn_W = tf.Variable(tf.random_normal((size_layer, np.unique(y_train).shape[0])))
    rnn_B = tf.Variable(tf.random_normal([np.unique(y_train).shape[0]]))
    logits = tf.matmul(outputs[:, -1], rnn_W) + rnn_B
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    cost += sum(beta * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    COST, TEST_COST, ACC, TEST_ACC = [], [], [], []

    for i in range(50):
        train_acc, train_loss = 0, 0
        for n in range(0, (len(x_train) // batch_size) * batch_size, batch_size):
            batch_x = np.zeros((batch_size, seq_len, 1))
            for k in range(batch_size):
                tokens = x_train[n + k].split()[:seq_len]
                for no, text in enumerate(tokens[::-1]):
                    try:
                        batch_x[k, -1 - no, 0] = dictionary[text]
                    except:
                        batch_x[k, -1 - no, 0] = -1
            _, loss = sess.run(
                [optimizer, cost], feed_dict={X: batch_x, Y: y_train[n : n + batch_size, :]}
            )
            train_acc += sess.run(
                accuracy, feed_dict={X: batch_x, Y: y_train[n : n + batch_size, :]}
            )
            train_loss += loss

        batch_x = np.zeros((len(x_test), seq_len, 1))
        for k in range(len(x_test)):
            tokens = x_test[k].split()[:seq_len]
            for no, text in enumerate(tokens[::-1]):
                try:
                    batch_x[k, -1 - no, 0] = dictionary[text]
                except:
                    batch_x[k, -1 - no, 0] = -1
        results = sess.run([cost, accuracy], feed_dict={X: batch_x, Y: y_test})
        TEST_COST.append(results[0])
        TEST_ACC.append(results[1])
        train_loss /= len(x_train) // batch_size
        train_acc /= len(x_train) // batch_size
        ACC.append(train_acc)
        COST.append(train_loss)
    COST = np.array(COST).mean()
    TEST_COST = np.array(TEST_COST).mean()
    ACC = np.array(ACC).mean()
    TEST_ACC = np.array(TEST_ACC).mean()
    return COST, TEST_COST, ACC, TEST_ACC


# In[8]:


def generate_nn(num_hidden, size_layer, learning_rate, dropout_rate, beta, activation, seq_len):
    global accbest
    param = {
        "num_hidden": int(np.around(num_hidden)),
        "size_layer": int(np.around(size_layer)),
        "learning_rate": max(min(learning_rate, 1), 0.0001),
        "dropout_rate": max(min(dropout_rate, 0.99), 0),
        "beta": max(min(beta, 0.5), 0.000001),
        "activation": int(np.around(activation)),
        "seq_len": int(np.around(seq_len)),
    }
    print("\nSearch parameters %s" % (param), file=log_file)
    log_file.flush()
    learning_cost, valid_cost, learning_acc, valid_acc = neural_network(**param)
    print(
        "stop after 50 iteration with train cost %f, valid cost %f, train acc %f, valid acc %f"
        % (learning_cost, valid_cost, learning_acc, valid_acc)
    )
    if valid_acc > accbest:
        costbest = valid_acc
    return valid_acc


# In[9]:


log_file = open("nn-bayesian.log", "a")
accbest = 0.0
NN_BAYESIAN = BayesianOptimization(
    generate_nn,
    {
        "num_hidden": (2, 10),
        "size_layer": (32, 512),
        "learning_rate": (0.0001, 1),
        "dropout_rate": (0.1, 0.99),
        "beta": (0.000001, 0.49),
        "activation": (0, 2),
        "seq_len": (5, 20),
    },
)
NN_BAYESIAN.maximize(init_points=10, n_iter=20, acq="ei", xi=0.0)


# In[10]:


print("Maximum NN accuracy value: %f" % NN_BAYESIAN.res["max"]["max_val"])
print("Best NN parameters: ", NN_BAYESIAN.res["max"]["max_params"])


# That means, best optimized parameters are:
# ```text
# num hidden: 2
# dropout rate: 0.16724952060867815
# beta: 0.099189911765081795
# learning rate: 0.049638440024850142
# size layer: 421
# activation: tanh
# sequence length: 20
# ```

# In[ ]:
