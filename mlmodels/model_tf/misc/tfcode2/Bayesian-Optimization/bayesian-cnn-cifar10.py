#!/usr/bin/env python
# coding: utf-8

# Please use Python 2. Python 3 got Cpickle problem

# In[1]:


import cPickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split

from bayes_opt import BayesianOptimization

# In[2]:


def unpickle(file):
    with open(file, "rb") as fo:
        dict = cPickle.load(fo)
    return dict


def reshape_image(img):
    img = img.reshape([3, 32, 32])
    return img.transpose([1, 2, 0])


unique_name = unpickle("/home/husein/space/cifar/cifar-10-batches-py/batches.meta")["label_names"]
cifar10 = unpickle("/home/husein/space/cifar/cifar-10-batches-py/data_batch_1")


# In[3]:


x_data = cifar10["data"][:300, :]
y_data = cifar10["labels"][:300]
onehot = np.zeros((x_data.shape[0], len(unique_name)))
for i in range(x_data.shape[0]):
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
# 1- batch size : 10
# 2- epoch: 20
# 3- adaptive gradient descent
# 4- softmax with cross entropy
# 5- 2 fully connected layers
# ```
#
# So you can change anything you want

# In[4]:


def neural_network(
    fully_conn_size,
    len_layer_conv,
    kernel_size,
    learning_rate,
    pooling_size,
    multiply,
    dropout_rate,
    beta,
    activation,
    batch_normalization,
    batch_size=10,
):

    tf.reset_default_graph()
    if activation == 0:
        activation = tf.nn.sigmoid
    elif activation == 1:
        activation = tf.nn.tanh
    else:
        activation = tf.nn.relu

    def conv_layer(x, conv, out_shape):
        w = tf.Variable(tf.truncated_normal([conv, conv, int(x.shape[3]), out_shape]))
        b = tf.Variable(tf.truncated_normal([out_shape], stddev=0.01))
        return tf.nn.conv2d(x, w, [1, 1, 1, 1], padding="SAME") + b

    def fully_connected(x, out_shape):
        w = tf.Variable(tf.truncated_normal([int(x.shape[1]), out_shape]))
        b = tf.Variable(tf.truncated_normal([out_shape], stddev=0.01))
        return tf.matmul(x, w) + b

    def pooling(x, k=2, stride=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding="SAME")

    X = tf.placeholder(tf.float32, (None, 32, 32, 3))
    Y = tf.placeholder(tf.float32, (None, len(unique_name)))
    train = tf.placeholder(tf.bool)
    for i in range(len_layer_conv):
        if i == 0:
            conv = activation(
                conv_layer(X, kernel_size, int(np.around(int(X.shape[3]) * multiply)))
            )
        else:
            conv = activation(
                conv_layer(conv, kernel_size, int(np.around(int(conv.shape[3]) * multiply)))
            )
        conv = pooling(conv, k=pooling_size, stride=pooling_size)
        if batch_normalization:
            conv = tf.layers.batch_normalization(conv, training=train)
        conv = tf.nn.dropout(conv, dropout_rate)
    print(conv.shape)
    output_shape = int(conv.shape[1]) * int(conv.shape[2]) * int(conv.shape[3])
    conv = tf.reshape(conv, [-1, output_shape])
    for i in range(2):
        if i == 0:
            fc = activation(fully_connected(conv, fully_conn_size))
        else:
            fc = activation(fully_connected(fc, fully_conn_size))
        fc = tf.nn.dropout(fc, dropout_rate)
        if batch_normalization:
            fc = tf.layers.batch_normalization(fc, training=train)
    logits = fully_connected(fc, len(unique_name))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
    cost += sum(beta * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    COST, TEST_COST, ACC, TEST_ACC = [], [], [], []
    for i in range(20):
        train_acc, train_loss = 0, 0
        for n in range(0, (x_train.shape[0] // batch_size) * batch_size, batch_size):
            batch_x = np.zeros((batch_size, 32, 32, 3))
            for k in range(batch_size):
                batch_x[k, :, :, :] = reshape_image(x_train[n + k, :])
            _, loss = sess.run(
                [optimizer, cost],
                feed_dict={X: batch_x, Y: y_train[n : n + batch_size, :], train: True},
            )
            train_acc += sess.run(
                accuracy, feed_dict={X: batch_x, Y: y_train[n : n + batch_size, :], train: False}
            )
            train_loss += loss
        batch_x = np.zeros((x_test.shape[0], 32, 32, 3))
        for k in range(x_test.shape[0]):
            batch_x[k, :, :, :] = reshape_image(x_test[k, :])
        results = sess.run([cost, accuracy], feed_dict={X: batch_x, Y: y_test, train: False})
        TEST_COST.append(results[0])
        TEST_ACC.append(results[1])
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


def generate_nn(
    fully_conn_size,
    len_layer_conv,
    kernel_size,
    learning_rate,
    pooling_size,
    multiply,
    dropout_rate,
    beta,
    activation,
    batch_normalization,
):
    global accbest
    param = {
        "fully_conn_size": int(np.around(fully_conn_size)),
        "len_layer_conv": int(np.around(len_layer_conv)),
        "kernel_size": int(np.around(kernel_size)),
        "learning_rate": max(min(learning_rate, 1), 0.0001),
        "pooling_size": int(np.around(pooling_size)),
        "multiply": multiply,
        "dropout_rate": max(min(dropout_rate, 0.99), 0),
        "beta": max(min(beta, 0.5), 0.000001),
        "activation": int(np.around(activation)),
        "batch_normalization": int(np.around(batch_normalization)),
    }
    learning_cost, valid_cost, learning_acc, valid_acc = neural_network(**param)
    print(
        "stop after 20 iteration with train cost %f, valid cost %f, train acc %f, valid acc %f"
        % (learning_cost, valid_cost, learning_acc, valid_acc)
    )
    if valid_acc > accbest:
        costbest = valid_acc
    return valid_acc


# In[6]:


accbest = 0.0
NN_BAYESIAN = BayesianOptimization(
    generate_nn,
    {
        "fully_conn_size": (16, 128),
        "len_layer_conv": (3, 5),
        "kernel_size": (2, 7),
        "learning_rate": (0.0001, 1),
        "pooling_size": (2, 4),
        "multiply": (1, 3),
        "dropout_rate": (0.1, 0.99),
        "beta": (0.000001, 0.49),
        "activation": (0, 2),
        "batch_normalization": (0, 1),
    },
)
NN_BAYESIAN.maximize(init_points=20, n_iter=40, acq="ei", xi=0.0)


# In[7]:


print("Maximum NN accuracy value: %f" % NN_BAYESIAN.res["max"]["max_val"])
print("Best NN parameters: %s" % NN_BAYESIAN.res["max"]["max_params"])


# In[ ]:
