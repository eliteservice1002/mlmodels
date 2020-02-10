#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import time

import tensorflow as tf
from sklearn.cross_validation import train_test_split
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


size_layer = 128
dimension_output = len(trainset.target_names)
maxlen = 50
batch_size = 32


# In[6]:


class Attention:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.dense_layer = tf.layers.Dense(hidden_size)
        self.v = tf.random_normal([hidden_size], mean=0, stddev=1 / np.sqrt(hidden_size))

    def score(self, hidden_tensor, encoder_outputs):
        energy = tf.nn.tanh(self.dense_layer(tf.concat([hidden_tensor, encoder_outputs], 2)))
        energy = tf.transpose(energy, [0, 2, 1])
        batch_size = tf.shape(encoder_outputs)[0]
        v = tf.expand_dims(tf.tile(tf.expand_dims(self.v, 0), [batch_size, 1]), 1)
        energy = tf.matmul(v, energy)
        return tf.squeeze(energy, 1)

    def __call__(self, hidden, encoder_outputs):
        seq_len = tf.shape(encoder_outputs)[1]
        batch_size = tf.shape(encoder_outputs)[0]
        H = tf.tile(tf.expand_dims(hidden, 1), [1, seq_len, 1])
        attn_energies = self.score(H, encoder_outputs)
        return tf.expand_dims(tf.nn.softmax(attn_energies), 1)


class Model:
    def __init__(
        self,
        dict_size,
        size_layers,
        learning_rate,
        num_classes,
        maxlen,
        num_blocks=3,
        block_size=128,
    ):
        self.X = tf.placeholder(tf.int32, [None, maxlen])
        self.Y = tf.placeholder(tf.int32, [None])
        embeddings = tf.Variable(tf.random_uniform([dict_size, size_layers], -1, 1))
        embedded = tf.nn.embedding_lookup(embeddings, self.X)
        self.attention = Attention(size_layers)

        def residual_block(x, size, rate, block):
            with tf.variable_scope("block_%d_%d" % (block, rate), reuse=False):
                attn_weights = self.attention(tf.reduce_sum(x, axis=1), x)
                conv_filter = tf.layers.conv1d(
                    attn_weights,
                    x.shape[2] // 4,
                    kernel_size=size,
                    strides=1,
                    padding="same",
                    dilation_rate=rate,
                    activation=tf.nn.tanh,
                )
                conv_gate = tf.layers.conv1d(
                    x,
                    x.shape[2] // 4,
                    kernel_size=size,
                    strides=1,
                    padding="same",
                    dilation_rate=rate,
                    activation=tf.nn.sigmoid,
                )
                out = tf.multiply(conv_filter, conv_gate)
                out = tf.layers.conv1d(
                    out, block_size, kernel_size=1, strides=1, padding="same", activation=tf.nn.tanh
                )
                return tf.add(x, out), out

        forward = tf.layers.conv1d(embedded, block_size, kernel_size=1, strides=1, padding="SAME")
        zeros = tf.zeros_like(forward)
        for i in range(num_blocks):
            for r in [1, 2, 4, 8, 16]:
                forward, s = residual_block(forward, size=7, rate=r, block=i)
                zeros = tf.add(zeros, s)
        self.logits = tf.reduce_sum(
            tf.layers.conv1d(forward, num_classes, kernel_size=1, strides=1, padding="SAME"), 1
        )
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1, output_type=tf.int32), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[7]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(len(dictionary), size_layer, 1e-3, dimension_output, maxlen)
sess.run(tf.global_variables_initializer())


# In[8]:


vectors = str_idx(trainset.data, dictionary, maxlen)
train_X, test_X, train_Y, test_Y = train_test_split(vectors, trainset.target, test_size=0.2)


# In[9]:


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
        batch_x_expand = np.expand_dims(batch_x, axis=1)
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
        batch_x_expand = np.expand_dims(batch_x, axis=1)
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


# In[10]:


real_Y, predict_Y = [], []

pbar = tqdm(range(0, len(test_X), batch_size), desc="validation minibatch loop")
for i in pbar:
    batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
    batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
    predict_Y += np.argmax(
        sess.run(model.logits, feed_dict={model.X: batch_x, model.Y: batch_y}), 1
    ).tolist()
    real_Y += batch_y


# In[11]:


print(metrics.classification_report(real_Y, predict_Y, target_names=trainset.target_names))


# In[ ]:
