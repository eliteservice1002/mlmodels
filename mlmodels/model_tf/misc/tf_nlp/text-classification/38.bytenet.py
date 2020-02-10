#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
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


def layer_normalization(x, epsilon=1e-8):
    shape = x.get_shape()
    tf.Variable(tf.zeros(shape=[int(shape[-1])]))
    beta = tf.Variable(tf.zeros(shape=[int(shape[-1])]))
    gamma = tf.Variable(tf.ones(shape=[int(shape[-1])]))
    mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)
    x = (x - mean) / tf.sqrt(variance + epsilon)
    return gamma * x + beta


def conv1d(input_, output_channels, dilation=1, filter_width=1, causal=False):
    w = tf.Variable(
        tf.random_normal(
            [1, filter_width, int(input_.get_shape()[-1]), output_channels], stddev=0.02
        )
    )
    b = tf.Variable(tf.zeros(shape=[output_channels]))
    if causal:
        padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]]
        padded = tf.pad(input_, padding)
        input_expanded = tf.expand_dims(padded, dim=1)
        out = tf.nn.atrous_conv2d(input_expanded, w, rate=dilation, padding="VALID") + b
    else:
        input_expanded = tf.expand_dims(input_, dim=1)
        out = tf.nn.atrous_conv2d(input_expanded, w, rate=dilation, padding="SAME") + b
    return tf.squeeze(out, [1])


def bytenet_residual_block(
    input_, dilation, layer_no, residual_channels, filter_width, causal=True
):
    block_type = "decoder" if causal else "encoder"
    block_name = "bytenet_{}_layer_{}_{}".format(block_type, layer_no, dilation)
    with tf.variable_scope(block_name):
        relu1 = tf.nn.relu(layer_normalization(input_))
        conv1 = conv1d(relu1, residual_channels)
        relu2 = tf.nn.relu(layer_normalization(conv1))
        dilated_conv = conv1d(relu2, residual_channels, dilation, filter_width, causal=causal)
        relu3 = tf.nn.relu(layer_normalization(dilated_conv))
        conv2 = conv1d(relu3, 2 * residual_channels)
        return input_ + conv2


class ByteNet:
    def __init__(
        self,
        dict_size,
        channels,
        encoder_dilations,
        dimension_output,
        encoder_filter_width,
        learning_rate=0.001,
        beta1=0.5,
    ):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, dimension_output])
        embedding_channels = 2 * channels
        w_source_embedding = tf.Variable(tf.random_uniform([dict_size, embedding_channels], -1, 1))
        source_embedding = tf.nn.embedding_lookup(w_source_embedding, self.X)
        for layer_no, dilation in enumerate(encoder_dilations):
            source_embedding = bytenet_residual_block(
                source_embedding, dilation, layer_no, channels, encoder_filter_width, causal=False
            )
        self.logits = conv1d(tf.nn.relu(source_embedding), dimension_output)[:, -1]
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[7]:


residual_channels = 256
encoder_dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
encoder_filter_width = 3
dimension_output = len(trainset.target_names)
batch_size = 128
maxlen = 50


# In[8]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = ByteNet(
    vocabulary_size + 4,
    residual_channels,
    encoder_dilations,
    dimension_output,
    encoder_filter_width,
)
sess.run(tf.global_variables_initializer())


# In[9]:


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
            feed_dict={model.X: batch_x, model.Y: test_onehot[i : i + batch_size]},
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
