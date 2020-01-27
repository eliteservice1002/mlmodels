#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numbers
import time

import tensorflow as tf
from sklearn.cross_validation import train_test_split
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.framework import ops, tensor_shape, tensor_util
from tensorflow.python.ops import array_ops, math_ops, random_ops
from tqdm import tqdm

from utils import *

# In[2]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[3]:


concat = " ".join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print("vocab from size: %d" % (vocabulary_size))
print("Most common words", count[4:10])
print("Sample data", data[:10], [rev_dictionary[i] for i in data[:10]])


# In[4]:


GO = dictionary["GO"]
PAD = dictionary["PAD"]
EOS = dictionary["EOS"]
UNK = dictionary["UNK"]


# In[5]:


embedding_size = 256
maxlen = 100
batch_size = 16
learning_rate = 1e-3
num_layers = 2


# In[33]:


def zoneout(x, keep_prob, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout") as name:
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError(
                "keep_prob must be a scalar tensor or a float in the "
                "range (0, 1], got %g" % keep_prob
            )
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor
        ret.set_shape(x.get_shape())
        return 1.0 - ret


class QRNN_pooling(tf.nn.rnn_cell.RNNCell):
    def __init__(self, out_fmaps):
        self.__out_fmaps = out_fmaps

    @property
    def state_size(self):
        return self.__out_fmaps

    @property
    def output_size(self):
        return self.__out_fmaps

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "QRNN-pooling"):
            Z, F, O = tf.split(inputs, 3, 1)
            new_state = tf.multiply(F, state) + tf.multiply(tf.subtract(1.0, F), Z)
            output = tf.multiply(O, new_state)
            return output, new_state


class QRNN_layer(object):
    """ Quasi-Recurrent Neural Network Layer
        (cf. https://arxiv.org/abs/1611.01576)
    """

    def __init__(
        self,
        out_fmaps,
        fwidth=2,
        activation=tf.tanh,
        pool_type="fo",
        zoneout=0.1,
        infer=False,
        bias_init_val=None,
        name="QRNN",
    ):
        self.out_fmaps = out_fmaps
        self.activation = activation
        self.name = name
        self.pool_type = pool_type
        self.fwidth = fwidth
        self.out_fmaps = out_fmaps
        self.zoneout = zoneout
        self.bias_init_val = bias_init_val

    def __call__(self, input_):
        input_shape = input_.get_shape().as_list()
        batch_size = tf.shape(input_)[0]
        fwidth = self.fwidth
        out_fmaps = self.out_fmaps
        zoneout = self.zoneout
        with tf.variable_scope(self.name):
            Z, gates = self.convolution(input_, fwidth, out_fmaps, zoneout)
            T = tf.concat([Z] + gates, 2)
            pooling = QRNN_pooling(out_fmaps)
            self.initial_state = pooling.zero_state(batch_size=batch_size, dtype=tf.float32)
            H, last_C = tf.nn.dynamic_rnn(pooling, T, initial_state=self.initial_state)
            self.Z = Z
            return H, last_C

    def convolution(self, input_, filter_width, out_fmaps, zoneout_):
        in_shape = input_.get_shape()
        in_fmaps = in_shape[-1]
        num_gates = num_layers
        gates = []
        pinput = tf.pad(input_, [[0, 0], [filter_width - 1, 0], [0, 0]])
        with tf.variable_scope("convolutions"):
            Wz = tf.get_variable(
                "Wz",
                [filter_width, in_fmaps, out_fmaps],
                initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05),
            )
            z_a = tf.nn.conv1d(pinput, Wz, stride=1, padding="VALID")
            if self.bias_init_val is not None:
                bz = tf.get_variable("bz", [out_fmaps], initializer=tf.constant_initializer(0.0))
                z_a += bz

            z = self.activation(z_a)
            for gate_name in range(num_gates):
                Wg = tf.get_variable(
                    "W{}".format(gate_name),
                    [filter_width, in_fmaps, out_fmaps],
                    initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05),
                )
                g_a = tf.nn.conv1d(pinput, Wg, stride=1, padding="VALID")
                if self.bias_init_val is not None:
                    bg = tf.get_variable(
                        "b{}".format(gate_name),
                        [out_fmaps],
                        initializer=tf.constant_initializer(0.0),
                    )
                    g_a += bg
                g = tf.sigmoid(g_a)
                gates.append(g)
        return z, gates


# In[36]:


class Model:
    def __init__(self):
        self.X = tf.placeholder(tf.int32, [None, maxlen])
        self.Y = tf.placeholder(tf.int32, [None])

        self.initial_states = []
        self.last_states = []
        self.qrnns = []
        with tf.variable_scope("QRNN_LM"):
            word_W = tf.get_variable(
                "word_W",
                [len(dictionary), embedding_size],
                initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05),
            )
            words = tf.split(1, maxlen, tf.expand_dims(self.X, -1))
            embeddings = tf.nn.embedding_lookup(word_W, self.X)

            qrnn_h = embeddings
            for qrnn_l in range(num_layers):
                qrnn_ = QRNN_layer(
                    embedding_size, pool_type="fo", zoneout=0.1, name="QRNN_layer{}".format(qrnn_l)
                )
                qrnn_h, last_state = qrnn_(qrnn_h)
                self.last_states.append(last_state)
                self.initial_states.append(qrnn_.initial_state)
                self.qrnns.append(qrnn_)
        self.logits = tf.layers.dense(qrnn_h[:, -1], len(trainset.target))

        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1, output_type=tf.int32), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[37]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())


# In[38]:


vectors = str_idx(trainset.data, dictionary, maxlen)
train_X, test_X, train_Y, test_Y = train_test_split(vectors, trainset.target, test_size=0.2)


# In[39]:


EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 3, 0, 0, 0

while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:%d\n" % (EPOCH))
        break

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    pbar = tqdm(range(0, len(train_X), batch_size), desc="train minibatch loop")
    for i in pbar:
        batch_x = train_X[i : min(i + batch_size, train_X.shape[0])]
        batch_y = train_Y[i : min(i + batch_size, train_X.shape[0])]
        acc, cost, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.Y: batch_y, model.X: batch_x},
        )
        assert not np.isnan(cost)
        train_loss += cost
        train_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    pbar = tqdm(range(0, len(test_X), batch_size), desc="test minibatch loop")
    for i in pbar:
        batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
        batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
        acc, cost = sess.run(
            [model.accuracy, model.cost], feed_dict={model.Y: batch_y, model.X: batch_x}
        )
        test_loss += cost
        test_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    train_loss /= len(train_X) / batch_size
    train_acc /= len(train_X) / batch_size
    test_loss /= len(test_X) / batch_size
    test_acc /= len(test_X) / batch_size

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


# In[40]:


real_Y, predict_Y = [], []

pbar = tqdm(range(0, len(test_X), batch_size), desc="validation minibatch loop")
for i in pbar:
    batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
    batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
    predict_Y += np.argmax(
        sess.run(model.logits, feed_dict={model.X: batch_x, model.Y: batch_y}), 1
    ).tolist()
    real_Y += batch_y


# In[41]:


print(metrics.classification_report(real_Y, predict_Y, target_names=trainset.target_names))


# In[ ]:
