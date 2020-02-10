#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

from bayes_opt import BayesianOptimization

# You can install Bayesian Optimization by,
# ```bash
# pip install bayesian-optimization
# ```

# In[2]:


df = pd.read_csv("Iris.csv")
df.head()


# In[3]:


x_data = df.iloc[:, :-1].values.astype(np.float32)
y_datalabel = df.iloc[:, -1]
y_data = LabelEncoder().fit_transform(df.iloc[:, -1])

onehot = np.zeros((y_data.shape[0], np.unique(y_data).shape[0]))
for i in range(y_data.shape[0]):
    onehot[i, y_data[i]] = 1.0

x_train, x_test, y_train, y_test, y_train_label, y_test_label = train_test_split(
    x_data, onehot, y_data, test_size=0.2
)


# ```text
# Activation function:
# 0- for sigmoid
# 1- for tanh
# 2- for relu
#
# Now the constants are:
# 1- batch size : 16
# 2- epoch: 100
# 3- gradient descent
# 4- softmax with cross entropy
# ```
#
# So you can change anything you want

# In[4]:


def neural_network(
    num_hidden, size_layer, learning_rate, dropout_rate, beta, activation, batch_size=16
):
    def activate(activation, first_layer, second_layer, bias):
        if activation == 0:
            activation = tf.nn.sigmoid
        elif activation == 1:
            activation = tf.nn.tanh
        else:
            activation = tf.nn.relu
        layer = activation(tf.matmul(first_layer, second_layer) + bias)
        return tf.nn.dropout(layer, dropout_rate)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, x_data.shape[1]))
    Y = tf.placeholder(tf.float32, (None, onehot.shape[1]))
    input_layer = tf.Variable(tf.random_normal([x_data.shape[1], size_layer]))
    biased_layer = tf.Variable(tf.random_normal([size_layer], stddev=0.1))
    output_layer = tf.Variable(tf.random_normal([size_layer, onehot.shape[1]]))
    biased_output = tf.Variable(tf.random_normal([onehot.shape[1]], stddev=0.1))
    layers, biased = [], []
    for i in range(num_hidden - 1):
        layers.append(tf.Variable(tf.random_normal([size_layer, size_layer])))
        biased.append(tf.Variable(tf.random_normal([size_layer])))
    first_l = activate(activation, X, input_layer, biased_layer)
    next_l = activate(activation, first_l, layers[0], biased[0])
    for i in range(1, num_hidden - 1):
        next_l = activate(activation, next_l, layers[i], biased[i])
    last_l = tf.matmul(next_l, output_layer) + biased_output
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=last_l, labels=Y))
    regularizers = (
        tf.nn.l2_loss(input_layer)
        + sum(map(lambda x: tf.nn.l2_loss(x), layers))
        + tf.nn.l2_loss(output_layer)
    )
    cost = cost + beta * regularizers
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(last_l, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    COST, TEST_COST, ACC, TEST_ACC = [], [], [], []
    for i in range(100):
        train_acc, train_loss = 0, 0
        for n in range(0, (x_train.shape[0] // batch_size) * batch_size, batch_size):
            _, loss = sess.run(
                [optimizer, cost],
                feed_dict={X: x_train[n : n + batch_size, :], Y: y_train[n : n + batch_size, :]},
            )
            train_acc += sess.run(
                accuracy,
                feed_dict={X: x_train[n : n + batch_size, :], Y: y_train[n : n + batch_size, :]},
            )
            train_loss += loss
        TEST_COST.append(sess.run(cost, feed_dict={X: x_test, Y: y_test}))
        TEST_ACC.append(sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
        train_loss /= x_train.shape[0] // batch_size
        train_acc /= x_train.shape[0] // batch_size
        ACC.append(train_acc)
        COST.append(train_loss)
    COST = np.array(COST).mean()
    TEST_COST = np.array(TEST_COST).mean()
    ACC = np.array(ACC).mean()
    TEST_ACC = np.array(TEST_ACC).mean()
    return COST, TEST_COST, ACC, TEST_ACC


# In[5]:


def generate_nn(num_hidden, size_layer, learning_rate, dropout_rate, beta, activation):
    global accbest
    param = {
        "num_hidden": int(np.around(num_hidden)),
        "size_layer": int(np.around(size_layer)),
        "learning_rate": max(min(learning_rate, 1), 0.0001),
        "dropout_rate": max(min(dropout_rate, 0.99), 0),
        "beta": max(min(beta, 0.5), 0.000001),
        "activation": int(np.around(activation)),
    }
    print("\nSearch parameters %s" % (param), file=log_file)
    log_file.flush()
    learning_cost, valid_cost, learning_acc, valid_acc = neural_network(**param)
    print(
        "stop after 200 iteration with train cost %f, valid cost %f, train acc %f, valid acc %f"
        % (learning_cost, valid_cost, learning_acc, valid_acc)
    )
    if valid_acc > accbest:
        costbest = valid_acc
    return valid_acc


# ```text
# hidden layers (2, 20)
# layer size (32, 1024)
# learning rate (0.0001, 1)
# dropout rate (0.1, 0.99)
# beta (0.000001, 0.49)
# activation (0, 2)
# ```
#
# You can set your own minimum and maximum boundaries, just change the value

# In[6]:


log_file = open("nn-bayesian.log", "a")
accbest = 0.0
NN_BAYESIAN = BayesianOptimization(
    generate_nn,
    {
        "num_hidden": (2, 20),
        "size_layer": (32, 1024),
        "learning_rate": (0.0001, 1),
        "dropout_rate": (0.1, 0.99),
        "beta": (0.000001, 0.49),
        "activation": (0, 2),
    },
)
NN_BAYESIAN.maximize(init_points=30, n_iter=50, acq="ei", xi=0.0)


# In[7]:


print("Maximum NN accuracy value: %f" % NN_BAYESIAN.res["max"]["max_val"])
print("Best NN parameters: ", NN_BAYESIAN.res["max"]["max_params"])


# So that means, best optimized parameters are:
# ```text
# dropout rate: 0.98999999999999999
# beta: 9.9999999999999995e-07
# learning rate: 0.0001
# size layer: 979 wide
# activation function: relu
# hidden layers: 2
# ```
