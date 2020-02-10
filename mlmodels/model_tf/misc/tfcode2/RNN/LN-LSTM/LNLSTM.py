#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import os
import pickle
import time

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split

# In[2]:


def layer_norm_all(h, base, num_units, scope):
    with tf.variable_scope(scope):
        h_reshape = tf.reshape(h, [-1, base, num_units])
        mean = tf.reduce_mean(h_reshape, [2], keep_dims=True)
        var = tf.reduce_mean(tf.square(h_reshape - mean), [2], keep_dims=True)
        epsilon = tf.constant(1e-3)
        rstd = tf.rsqrt(var + epsilon)
        h_reshape = (h_reshape - mean) * rstd
        h = tf.reshape(h_reshape, [-1, base * num_units])
        alpha = tf.get_variable(
            "layer_norm_alpha",
            [4 * num_units],
            initializer=tf.constant_initializer(1.0),
            dtype=tf.float32,
        )
        bias = tf.get_variable(
            "layer_norm_bias",
            [4 * num_units],
            initializer=tf.constant_initializer(0.0),
            dtype=tf.float32,
        )
        return (h * alpha) + bias


def layer_norm(x, scope="layer_norm", alpha_start=1.0, bias_start=0.0):
    with tf.variable_scope(scope):
        num_units = x.get_shape().as_list()[1]
        alpha = tf.get_variable(
            "alpha", [num_units], initializer=tf.constant_initializer(alpha_start), dtype=tf.float32
        )
        bias = tf.get_variable(
            "bias", [num_units], initializer=tf.constant_initializer(bias_start), dtype=tf.float32
        )
        mean, variance = moments_for_layer_norm(x)
        y = (alpha * (x - mean)) / (variance) + bias
    return y


def moments_for_layer_norm(x, axes=1, name=None):
    epsilon = 1e-3
    if not isinstance(axes, list):
        axes = [axes]
    mean = tf.reduce_mean(x, axes, keep_dims=True)
    variance = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axes, keep_dims=True) + epsilon)
    return mean, variance


def zoneout(new_h, new_c, h, c, h_keep, c_keep, is_training):
    mask_c = tf.ones_like(c)
    mask_h = tf.ones_like(h)
    if is_training:
        mask_c = tf.nn.dropout(mask_c, c_keep)
        mask_h = tf.nn.dropout(mask_h, h_keep)
    mask_c *= c_keep
    mask_h *= h_keep
    h = new_h * mask_h + (-mask_h + 1.0) * h
    c = new_c * mask_c + (-mask_c + 1.0) * c
    return h, c


class LN_LSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(
        self,
        num_units,
        f_bias=1.0,
        use_zoneout=False,
        zoneout_keep_h=0.9,
        zoneout_keep_c=0.5,
        is_training=True,
        reuse=None,
        name=None,
    ):
        super(LN_LSTMCell, self).__init__(_reuse=reuse, name=name)
        self.num_units = num_units
        self.f_bias = f_bias
        self.use_zoneout = use_zoneout
        self.zoneout_keep_h = zoneout_keep_h
        self.zoneout_keep_c = zoneout_keep_c
        self.is_training = is_training

    def build(self, inputs_shape):
        w_init = tf.orthogonal_initializer(1.0)
        h_init = tf.orthogonal_initializer(1.0)
        b_init = tf.constant_initializer(0.0)
        h_size = self.num_units
        self.W_xh = tf.get_variable(
            "W_xh", [inputs_shape[1], 4 * h_size], initializer=w_init, dtype=tf.float32
        )
        self.W_hh = tf.get_variable(
            "W_hh", [h_size, 4 * h_size], initializer=h_init, dtype=tf.float32
        )
        self.bias = tf.get_variable("bias", [4 * h_size], initializer=b_init, dtype=tf.float32)

    def call(self, x, state):
        h, c = state
        h_size = self.num_units
        concat = tf.concat(axis=1, values=[x, h])
        W_full = tf.concat(axis=0, values=[self.W_xh, self.W_hh])
        concat = tf.matmul(concat, W_full) + self.bias
        concat = layer_norm_all(concat, 4, h_size, "ln")
        i, j, f, o = tf.split(axis=1, num_or_size_splits=4, value=concat)
        new_c = c * tf.sigmoid(f + self.f_bias) + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.tanh(layer_norm(new_c, "ln_c")) * tf.sigmoid(o)
        if self.use_zoneout:
            new_h, new_c = zoneout(
                new_h, new_c, h, c, self.zoneout_keep_h, self.zoneout_keep_c, self.is_training
            )
        return new_h, new_c

    def zero_state(self, batch_size, dtype):
        h = tf.zeros([batch_size, self.num_units], dtype=dtype)
        c = tf.zeros([batch_size, self.num_units], dtype=dtype)
        return (h, c)

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units


# In[3]:


class Model:
    def __init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate):
        def lstm_cell():
            return tf.contrib.rnn.LayerNormBasicLSTMCell(size_layer)

        self.rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
        self.X = tf.placeholder(tf.float32, [None, None, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        drop = tf.contrib.rnn.DropoutWrapper(self.rnn_cells, output_keep_prob=0.5)
        self.outputs, self.last_state = tf.nn.dynamic_rnn(drop, self.X, dtype=tf.float32)
        self.rnn_W = tf.Variable(tf.random_normal((size_layer, dimension_output)))
        self.rnn_B = tf.Variable(tf.random_normal([dimension_output]))
        self.logits = tf.matmul(self.outputs[:, -1], self.rnn_W) + self.rnn_B
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        l2 = sum(0.0005 * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        self.cost += l2
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


# In[4]:


maxlen = 20
location = os.getcwd()
num_layers = 2
size_layer = 256
learning_rate = 1e-7
batch = 100


# In[5]:


with open("dataset-emotion.p", "rb") as fopen:
    df = pickle.load(fopen)
with open("vector-emotion.p", "rb") as fopen:
    vectors = pickle.load(fopen)
with open("dataset-dictionary.p", "rb") as fopen:
    dictionary = pickle.load(fopen)


# In[ ]:


label = np.unique(df[:, 1])
train_X, test_X, train_Y, test_Y = train_test_split(df[:, 0], df[:, 1].astype("int"), test_size=0.2)


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
        saver.save(sess, os.getcwd() + "/model-rnn-vector.ckpt")
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
