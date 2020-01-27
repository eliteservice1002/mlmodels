#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from IPython.display import HTML
from matplotlib import animation
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sns.set()


# In[2]:


df = pd.read_csv("Iris.csv")
df.head()


# In[3]:


X = PCA(n_components=2).fit_transform(MinMaxScaler().fit_transform(df.iloc[:, 1:-1]))
Y = LabelEncoder().fit_transform(df.iloc[:, -1])
onehot_y = np.zeros((X.shape[0], np.unique(Y).shape[0]))
for k in range(X.shape[0]):
    onehot_y[k, Y[k]] = 1.0


# In[4]:


class Model:
    def __init__(self, learning_rate, layer_size, optimizer):
        self.X = tf.placeholder(tf.float32, (None, X.shape[1]))
        self.Y = tf.placeholder(tf.float32, (None, np.unique(Y).shape[0]))
        w1 = tf.Variable(tf.random_normal([X.shape[1], layer_size]))
        b1 = tf.Variable(tf.random_normal([layer_size]))
        w2 = tf.Variable(tf.random_normal([layer_size, layer_size]))
        b2 = tf.Variable(tf.random_normal([layer_size]))
        w3 = tf.Variable(tf.random_normal([layer_size, np.unique(Y).shape[0]]))
        b3 = tf.Variable(tf.random_normal([np.unique(Y).shape[0]]))
        self.logits = tf.nn.sigmoid(tf.matmul(self.X, w1) + b1)
        self.logits = tf.nn.sigmoid(tf.matmul(self.logits, w2) + b2)
        self.logits = tf.matmul(self.logits, w3) + b3
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits)
        )
        self.optimizer = optimizer(learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[5]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(0.1, 128, tf.train.GradientDescentOptimizer)
sess.run(tf.global_variables_initializer())
for i in range(100):
    batch_y = np.zeros((X.shape[0], np.unique(Y).shape[0]))
    for k in range(X.shape[0]):
        batch_y[k, Y[k]] = 1.0
    cost, _ = sess.run([model.cost, model.optimizer], feed_dict={model.X: X, model.Y: onehot_y})
    acc = sess.run(model.accuracy, feed_dict={model.X: X, model.Y: onehot_y})
    if (i + 1) % 10 == 0:
        print("epoch %d, Entropy: %f, Accuracy: %f" % (i + 1, cost, acc))


# In[6]:


fig = plt.figure(figsize=(15, 10))
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = np.argmax(sess.run(model.logits, feed_dict={model.X: np.c_[xx.ravel(), yy.ravel()]}), axis=1)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
plt.show()


# In[7]:


tf.reset_default_graph()
first_graph = tf.Graph()
with first_graph.as_default():
    gd = Model(0.1, 128, tf.train.GradientDescentOptimizer)
    first_sess = tf.InteractiveSession()
    first_sess.run(tf.global_variables_initializer())

second_graph = tf.Graph()
with second_graph.as_default():
    adagrad = Model(0.1, 128, tf.train.AdagradOptimizer)
    second_sess = tf.InteractiveSession()
    second_sess.run(tf.global_variables_initializer())

third_graph = tf.Graph()
with third_graph.as_default():
    rmsprop = Model(0.1, 128, tf.train.RMSPropOptimizer)
    third_sess = tf.InteractiveSession()
    third_sess.run(tf.global_variables_initializer())

fourth_graph = tf.Graph()
with fourth_graph.as_default():
    adam = Model(0.1, 128, tf.train.AdamOptimizer)
    fourth_sess = tf.InteractiveSession()
    fourth_sess.run(tf.global_variables_initializer())


# In[8]:


fig = plt.figure(figsize=(25, 15))
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
concated = np.c_[xx.ravel(), yy.ravel()]
plt.subplot(2, 2, 1)
Z = first_sess.run(gd.logits, feed_dict={gd.X: concated})
acc = first_sess.run(gd.accuracy, feed_dict={gd.X: X, gd.Y: onehot_y})
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
plt.title("GD epoch %d, acc %f" % (0, acc))
plt.subplot(2, 2, 2)
Z = second_sess.run(adagrad.logits, feed_dict={adagrad.X: concated})
acc = second_sess.run(adagrad.accuracy, feed_dict={adagrad.X: X, adagrad.Y: onehot_y})
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
plt.title("ADAGRAD epoch %d, acc %f" % (0, acc))
plt.subplot(2, 2, 3)
Z = third_sess.run(rmsprop.logits, feed_dict={rmsprop.X: concated})
acc = third_sess.run(rmsprop.accuracy, feed_dict={rmsprop.X: X, rmsprop.Y: onehot_y})
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
plt.title("RMSPROP epoch %d, acc %f" % (0, acc))
plt.subplot(2, 2, 4)
Z = fourth_sess.run(adam.logits, feed_dict={adam.X: concated})
acc = fourth_sess.run(adam.accuracy, feed_dict={adam.X: X, adam.Y: onehot_y})
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
plt.title("ADAM epoch %d, acc %f" % (0, acc))


def training(epoch):
    plt.subplot(2, 2, 1)
    first_sess.run(gd.optimizer, feed_dict={gd.X: X, gd.Y: onehot_y})
    Z = first_sess.run(gd.logits, feed_dict={gd.X: concated})
    acc = first_sess.run(gd.accuracy, feed_dict={gd.X: X, gd.Y: onehot_y})
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
    plt.title("GD epoch %d, acc %f" % (epoch, acc))
    plt.subplot(2, 2, 2)
    second_sess.run(adagrad.optimizer, feed_dict={adagrad.X: X, adagrad.Y: onehot_y})
    Z = second_sess.run(adagrad.logits, feed_dict={adagrad.X: concated})
    acc = second_sess.run(adagrad.accuracy, feed_dict={adagrad.X: X, adagrad.Y: onehot_y})
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
    plt.title("ADAGRAD epoch %d, acc %f" % (epoch, acc))
    plt.subplot(2, 2, 3)
    third_sess.run(rmsprop.optimizer, feed_dict={rmsprop.X: X, rmsprop.Y: onehot_y})
    Z = third_sess.run(rmsprop.logits, feed_dict={rmsprop.X: concated})
    acc = third_sess.run(rmsprop.accuracy, feed_dict={rmsprop.X: X, rmsprop.Y: onehot_y})
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
    plt.title("RMSPROP epoch %d, acc %f" % (epoch, acc))
    plt.subplot(2, 2, 4)
    fourth_sess.run(adam.optimizer, feed_dict={adam.X: X, adam.Y: onehot_y})
    Z = fourth_sess.run(adam.logits, feed_dict={adam.X: concated})
    acc = fourth_sess.run(adam.accuracy, feed_dict={adam.X: X, adam.Y: onehot_y})
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
    cont = plt.title("ADAM epoch %d, acc %f" % (epoch, acc))
    return cont


anim = animation.FuncAnimation(fig, training, frames=100, interval=200)
anim.save("animation-gradientcomparison-iris.gif", writer="imagemagick", fps=5)


# In[ ]:
