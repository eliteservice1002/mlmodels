#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import constant_op, ops
from tensorflow.python.ops import control_flow_ops, math_ops, state_ops
from tensorflow.python.training import optimizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# In[2]:


class COCOB(optimizer.Optimizer):
    def __init__(self, alpha=100, use_locking=False, name="COCOB"):
        super(COCOB, self).__init__(use_locking, name)
        self._alpha = alpha

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                gradients_sum = constant_op.constant(
                    0, shape=v.get_shape(), dtype=v.dtype.base_dtype
                )
                grad_norm_sum = constant_op.constant(
                    0, shape=v.get_shape(), dtype=v.dtype.base_dtype
                )
                L = constant_op.constant(1e-8, shape=v.get_shape(), dtype=v.dtype.base_dtype)
                tilde_w = constant_op.constant(0.0, shape=v.get_shape(), dtype=v.dtype.base_dtype)
                reward = constant_op.constant(0.0, shape=v.get_shape(), dtype=v.dtype.base_dtype)
            self._get_or_make_slot(v, L, "L", self._name)
            self._get_or_make_slot(v, grad_norm_sum, "grad_norm_sum", self._name)
            self._get_or_make_slot(v, gradients_sum, "gradients_sum", self._name)
            self._get_or_make_slot(v, tilde_w, "tilde_w", self._name)
            self._get_or_make_slot(v, reward, "reward", self._name)

    def _apply_dense(self, grad, var):
        gradients_sum = self.get_slot(var, "gradients_sum")
        grad_norm_sum = self.get_slot(var, "grad_norm_sum")
        tilde_w = self.get_slot(var, "tilde_w")
        L = self.get_slot(var, "L")
        reward = self.get_slot(var, "reward")
        L_update = tf.maximum(L, tf.abs(grad))
        gradients_sum_update = gradients_sum + grad
        grad_norm_sum_update = grad_norm_sum + tf.abs(grad)
        reward_update = tf.maximum(reward - grad * tilde_w, 0)
        new_w = (
            -gradients_sum_update
            / (L_update * (tf.maximum(grad_norm_sum_update + L_update, self._alpha * L_update)))
            * (reward_update + L_update)
        )
        var_update = var - tilde_w + new_w
        tilde_w_update = new_w
        gradients_sum_update_op = state_ops.assign(gradients_sum, gradients_sum_update)
        grad_norm_sum_update_op = state_ops.assign(grad_norm_sum, grad_norm_sum_update)
        var_update_op = state_ops.assign(var, var_update)
        tilde_w_update_op = state_ops.assign(tilde_w, tilde_w_update)
        L_update_op = state_ops.assign(L, L_update)
        reward_update_op = state_ops.assign(reward, reward_update)
        return control_flow_ops.group(
            *[
                gradients_sum_update_op,
                var_update_op,
                grad_norm_sum_update_op,
                tilde_w_update_op,
                reward_update_op,
                L_update_op,
            ]
        )

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)


# In[3]:


mnist = input_data.read_data_sets("", validation_size=0)


# In[4]:


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
        self.optimizer = COCOB().minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[5]:


batch_size = 128
epoch = 10

train_images = mnist.train.images
test_images = mnist.test.images

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())


# In[6]:


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
