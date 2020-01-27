#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops, math_ops, state_ops
from tensorflow.python.training import optimizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# In[3]:


class AddSign(optimizer.Optimizer):
    def __init__(
        self, learning_rate=1.001, alpha=0.01, beta=0.5, use_locking=False, name="AddSign"
    ):
        super(AddSign, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._beta, name="beta_t")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
        eps = 1e-7
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))
        var_update = state_ops.assign_sub(
            var, lr_t * grad * (1.0 + alpha_t * tf.sign(grad) * tf.sign(m_t))
        )
        return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


# In[4]:


mnist = input_data.read_data_sets("", validation_size=0)


# In[5]:


class Model:
    def __init__(self, learning_rate=0.01):
        self.X = tf.placeholder(tf.float32, shape=[None, 784])
        self.Y = tf.placeholder(tf.float32, shape=[None, 10])
        w1 = tf.Variable(tf.random_normal([784, 200]))
        b1 = tf.Variable(tf.random_normal([200]))
        w2 = tf.Variable(tf.random_normal([200, 100]))
        b2 = tf.Variable(tf.random_normal([100]))
        w3 = tf.Variable(tf.random_normal([100, 10]))
        b3 = tf.Variable(tf.random_normal([10]))
        feedforward = tf.nn.relu(tf.matmul(self.X, w1) + b1)
        feedforward = tf.nn.relu(tf.matmul(feedforward, w2) + b2)
        self.logits = tf.matmul(feedforward, w3) + b3
        self.cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.logits)
        )
        self.optimizer = AddSign(learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[8]:


batch_size = 128
epoch = 10

train_images = mnist.train.images
test_images = mnist.test.images

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())


# In[11]:


LOSS, ACC_TRAIN, ACC_TEST = [], [], []
for i in range(epoch):
    total_loss, total_acc = 0, 0
    for n in range(0, (mnist.train.images.shape[0] // batch_size) * batch_size, batch_size):
        batch_x = train_images[n : n + batch_size, :]
        batch_y = np.zeros((batch_size, 10))
        batch_y[np.arange(batch_size), mnist.train.labels[n : n + batch_size]] = 1.0
        cost, _ = sess.run(
            [model.cost, model.optimizer], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        total_acc += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})
        total_loss += cost
    total_loss /= mnist.train.images.shape[0] // batch_size
    total_acc /= mnist.train.images.shape[0] // batch_size
    ACC_TRAIN.append(total_acc)
    total_acc = 0
    for n in range(
        0, (mnist.test.images[:1000, :].shape[0] // batch_size) * batch_size, batch_size
    ):
        batch_x = test_images[n : n + batch_size, :]
        batch_y = np.zeros((batch_size, 10))
        batch_y[np.arange(batch_size), mnist.test.labels[n : n + batch_size]] = 1.0
        total_acc += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})
    total_acc /= mnist.test.images[:1000, :].shape[0] // batch_size
    ACC_TEST.append(total_acc)
    print(
        "epoch: %d, accuracy train: %f, accuracy testing: %f" % (i + 1, ACC_TRAIN[-1], ACC_TEST[-1])
    )


# In[ ]:
