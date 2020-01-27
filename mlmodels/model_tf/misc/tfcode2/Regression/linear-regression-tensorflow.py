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

sns.set()
df = pd.read_csv("poverty.csv")
df.head()


# In[2]:


X = df.iloc[:, 1:2].values
Y = df.iloc[:, 2:3].values


# In[3]:


class Linear:
    def __init__(self, learning_rate):
        self.X = tf.placeholder(tf.float32, (None, 1))
        self.Y = tf.placeholder(tf.float32, (None, 1))
        w = tf.Variable(tf.random_normal([1, 1]))
        b = tf.Variable(tf.random_normal([1]))
        self.logits = tf.matmul(self.X, w) + b
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)


# In[4]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Linear(0.001)
sess.run(tf.global_variables_initializer())
for i in range(10):
    cost, _ = sess.run([model.cost, model.optimizer], feed_dict={model.X: X, model.Y: Y})
    print("epoch %d, MSE: %f" % (i + 1, cost))


# In[5]:


y_output = sess.run(model.logits, feed_dict={model.X: X})
plt.scatter(X[:, 0], Y[:, 0])
plt.plot(X, y_output, c="red")
plt.show()


# In[6]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Linear(0.001)
sess.run(tf.global_variables_initializer())

fig = plt.figure(figsize=(10, 5))
ax = plt.axes()
ax.scatter(X[:, 0], Y[:, 0], c="b")
cost, y_output = sess.run([model.cost, model.logits], feed_dict={model.X: X, model.Y: Y})
ax.set_xlabel("epoch: %d, MSE: %f" % (0, cost))
line, = ax.plot(X, y_output, lw=2, c="r")


def gradient_mean_square(epoch):
    cost, y_output, _ = sess.run(
        [model.cost, model.logits, model.optimizer], feed_dict={model.X: X, model.Y: Y}
    )
    line.set_data(X, y_output)
    ax.set_xlabel("epoch: %d, MSE: %f" % (epoch, cost))
    return line, ax


anim = animation.FuncAnimation(fig, gradient_mean_square, frames=50, interval=200)
anim.save("animation-linear-regression.gif", writer="imagemagick", fps=10)


# In[ ]:
