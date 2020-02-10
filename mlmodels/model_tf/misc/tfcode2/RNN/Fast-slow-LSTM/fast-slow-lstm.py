#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import tensorflow as tf
from sklearn.cross_validation import train_test_split

from utils import *

# In[2]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[3]:


ONEHOT = np.zeros((len(trainset.data), len(trainset.target_names)))
ONEHOT[np.arange(len(trainset.data)), trainset.target] = 1.0
train_X, test_X, train_Y, test_Y, train_onehot, test_onehot = train_test_split(
    trainset.data, trainset.target, ONEHOT, test_size=0.2
)


# In[4]:


concat = " ".join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print("vocab from size: %d" % (vocabulary_size))
print("Most common words", count[4:10])
print("Sample data", data[:10], [rev_dictionary[i] for i in data[:10]])


# In[5]:


GO = dictionary["GO"]
PAD = dictionary["PAD"]
EOS = dictionary["EOS"]
UNK = dictionary["UNK"]


# In[6]:


def moments_for_layer_norm(x, axes=1, name=None):
    epsilon = 1e-3
    if not isinstance(axes, list):
        axes = [axes]
    mean = tf.reduce_mean(x, axes, keep_dims=True)
    variance = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axes, keep_dims=True) + epsilon)
    return mean, variance


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
    return (alpha * (x - mean)) / (variance) + bias


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
        is_training=False,
    ):
        self.num_units = num_units
        self.f_bias = f_bias

        self.use_zoneout = use_zoneout
        self.zoneout_keep_h = zoneout_keep_h
        self.zoneout_keep_c = zoneout_keep_c

        self.is_training = is_training

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            h, c = state
            h_size = self.num_units
            x_size = x.get_shape().as_list()[1]
            w_init = tf.constant_initializer(1.0)
            h_init = tf.constant_initializer(1.0)
            b_init = tf.constant_initializer(0.0)
            W_xh = tf.get_variable(
                "W_xh", [x_size, 4 * h_size], initializer=w_init, dtype=tf.float32
            )
            W_hh = tf.get_variable(
                "W_hh", [h_size, 4 * h_size], initializer=h_init, dtype=tf.float32
            )
            bias = tf.get_variable("bias", [4 * h_size], initializer=b_init, dtype=tf.float32)
            concat = tf.concat(axis=1, values=[x, h])
            W_full = tf.concat(axis=0, values=[W_xh, W_hh])
            concat = tf.matmul(concat, W_full) + bias
            concat = layer_norm_all(concat, 4, h_size, "ln")
            i, j, f, o = tf.split(axis=1, num_or_size_splits=4, value=concat)
            new_c = c * tf.sigmoid(f + self.f_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(layer_norm(new_c, "ln_c")) * tf.sigmoid(o)
            if self.use_zoneout:
                new_h, new_c = zoneout(
                    new_h, new_c, h, c, self.zoneout_keep_h, self.zoneout_keep_c, self.is_training
                )
        return new_h, (new_h, new_c)

    def zero_state(self, batch_size, dtype):
        h = tf.zeros([batch_size, self.num_units], dtype=dtype)
        c = tf.zeros([batch_size, self.num_units], dtype=dtype)
        return (h, c)


class FSRNNCell(tf.contrib.rnn.RNNCell):
    def __init__(self, fast_cells, slow_cell, keep_prob=1.0, training=True):
        self.fast_layers = len(fast_cells)
        assert self.fast_layers >= 2, "At least two fast layers are needed"
        self.fast_cells = fast_cells
        self.slow_cell = slow_cell
        self.keep_prob = keep_prob
        if not training:
            self.keep_prob = 1.0

    def __call__(self, inputs, state, scope="FS-RNN"):
        F_state = state[0]
        S_state = state[1]

        with tf.variable_scope(scope):
            inputs = tf.nn.dropout(inputs, self.keep_prob)

            with tf.variable_scope("Fast_0"):
                F_output, F_state = self.fast_cells[0](inputs, F_state)
            F_output_drop = tf.nn.dropout(F_output, self.keep_prob)

            with tf.variable_scope("Slow"):
                S_output, S_state = self.slow_cell(F_output_drop, S_state)
            S_output_drop = tf.nn.dropout(S_output, self.keep_prob)

            with tf.variable_scope("Fast_1"):
                F_output, F_state = self.fast_cells[1](S_output_drop, F_state)

            for i in range(2, self.fast_layers):
                with tf.variable_scope("Fast_" + str(i)):
                    F_output, F_state = self.fast_cells[i](F_output[:, 0:1] * 0.0, F_state)

            F_output_drop = tf.nn.dropout(F_output, self.keep_prob)
            return F_output_drop, (F_state, S_state)

    def zero_state(self, batch_size, dtype):
        F_state = self.fast_cells[0].zero_state(batch_size, dtype)
        S_state = self.slow_cell.zero_state(batch_size, dtype)
        return (F_state, S_state)


# In[10]:


class Model:
    def __init__(
        self,
        size_layer,
        num_layers,
        fast_layers,
        embedded_size,
        dict_size,
        dimension_output,
        learning_rate,
        batch_size,
        timestamp,
        is_training=True,
        zoneout_h=0.95,
        zoneout_c=0.7,
        keep_prob=0.75,
    ):

        self.X = tf.placeholder(tf.int32, [batch_size, maxlen])
        self.Y = tf.placeholder(tf.float32, [batch_size, dimension_output])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)

        F_cells = [
            LN_LSTMCell(
                fast_layers,
                use_zoneout=True,
                is_training=is_training,
                zoneout_keep_h=zoneout_h,
                zoneout_keep_c=zoneout_c,
            )
            for _ in range(num_layers)
        ]
        S_cell = LN_LSTMCell(
            size_layer,
            use_zoneout=True,
            is_training=is_training,
            zoneout_keep_h=zoneout_h,
            zoneout_keep_c=zoneout_c,
        )
        FS_cell = FSRNNCell(F_cells, S_cell, keep_prob, is_training)
        self._initial_state = FS_cell.zero_state(batch_size, tf.float32)
        state = self._initial_state
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(timestamp):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                out, state = FS_cell(encoder_embedded[:, time_step, :], state)
                outputs.append(out)
        outputs = tf.reshape(tf.concat(outputs, axis=1), [batch_size, timestamp, fast_layer])
        W = tf.get_variable(
            "w", shape=(fast_layer, dimension_output), initializer=tf.orthogonal_initializer()
        )
        b = tf.get_variable("b", shape=(dimension_output), initializer=tf.zeros_initializer())
        self.logits = tf.matmul(outputs[:, -1], W) + b
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[8]:


size_layer = 64
fast_layer = 128
num_layers = 2
embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 128


# In[11]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(
    size_layer,
    num_layers,
    fast_layer,
    embedded_size,
    vocabulary_size + 4,
    dimension_output,
    learning_rate,
    batch_size,
    maxlen,
)
sess.run(tf.global_variables_initializer())


# In[12]:


EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:%d\n" % (EPOCH))
        break

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (len(train_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(train_X[i : i + batch_size], dictionary, maxlen)
        acc, loss, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X: batch_x, model.Y: train_onehot[i : i + batch_size]},
        )
        train_loss += loss
        train_acc += acc

    for i in range(0, (len(test_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(test_X[i : i + batch_size], dictionary, maxlen)
        acc, loss = sess.run(
            [model.accuracy, model.cost],
            feed_dict={model.X: batch_x, model.Y: train_onehot[i : i + batch_size]},
        )
        test_loss += loss
        test_acc += acc

    train_loss /= len(train_X) // batch_size
    train_acc /= len(train_X) // batch_size
    test_loss /= len(test_X) // batch_size
    test_acc /= len(test_X) // batch_size

    if test_acc > CURRENT_ACC:
        print("epoch: %d, pass acc: %f, current acc: %f" % (EPOCH, CURRENT_ACC, test_acc))
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
    else:
        CURRENT_CHECKPOINT += 1

    print("time taken:", time.time() - lasttime)
    print(
        "epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n"
        % (EPOCH, train_loss, train_acc, test_loss, test_acc)
    )
    EPOCH += 1


# In[ ]:
