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


def temporal_padding(x, padding=(1, 1)):
    return tf.pad(x, [[0, 0], [padding[0], padding[1]], [0, 0]])


def attention_block(x):
    k_size = x.get_shape()[-1].value
    v_size = x.get_shape()[-1].value
    key = tf.layers.dense(
        x,
        units=k_size,
        activation=None,
        use_bias=False,
        kernel_initializer=tf.random_normal_initializer(0, 0.01),
    )
    query = tf.layers.dense(
        x,
        units=v_size,
        activation=None,
        use_bias=False,
        kernel_initializer=tf.random_normal_initializer(0, 0.01),
    )
    logits = tf.matmul(key, key, transpose_b=True)
    logits = logits / np.sqrt(k_size)
    weights = tf.nn.softmax(logits, name="attention_weights")
    return tf.matmul(weights, query)


def convolution1d(x, num_filters, dilation_rate, k, filter_size=3, stride=[1], pad="VALID"):
    with tf.variable_scope("conv1d_%d" % (k)):
        num_filters = num_filters * 2
        V = tf.get_variable(
            "V",
            [filter_size, int(x.get_shape()[-1]), num_filters],
            tf.float32,
            initializer=None,
            trainable=True,
        )
        g = tf.get_variable(
            "g",
            shape=[num_filters],
            dtype=tf.float32,
            initializer=tf.constant_initializer(1.0),
            trainable=True,
        )
        b = tf.get_variable(
            "b", shape=[num_filters], dtype=tf.float32, initializer=None, trainable=True
        )
        W = tf.reshape(g, [1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1])
        left_pad = dilation_rate * (filter_size - 1)
        x = temporal_padding(x, (left_pad, 0))
        x = tf.nn.bias_add(tf.nn.convolution(x, W, pad, stride, [dilation_rate]), b)
        split0, split1 = tf.split(x, num_or_size_splits=2, axis=2)
        split1 = tf.sigmoid(split1)
        return tf.multiply(split0, split1)


def temporalblock(
    input_layer, out_channels, filter_size, stride, dilation_rate, dropout, k, highway=False
):
    keep_prob = 1.0 - dropout
    in_channels = input_layer.get_shape()[-1]
    count = 0
    with tf.variable_scope("temporal_block_%d" % (k)):
        conv1 = convolution1d(
            input_layer, out_channels, dilation_rate, count, filter_size, [stride]
        )
        noise_shape = (tf.shape(conv1)[0], tf.constant(1), tf.shape(conv1)[2])
        dropout1 = tf.nn.dropout(conv1, keep_prob, noise_shape)
        dropout1 = attention_block(dropout1)
        count += 1
        conv2 = convolution1d(
            input_layer, out_channels, dilation_rate, count, filter_size, [stride]
        )
        dropout2 = tf.nn.dropout(conv2, keep_prob, noise_shape)
        dropout2 = attention_block(dropout2)
        residual = None
        if highway:
            W_h = tf.get_variable(
                "W_h",
                [1, int(input_layer.get_shape()[-1]), out_channels],
                tf.float32,
                tf.random_normal_initializer(0, 0.01),
                trainable=True,
            )
            b_h = tf.get_variable(
                "b_h", shape=[out_channels], dtype=tf.float32, initializer=None, trainable=True
            )
            H = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, "SAME"), b_h)
            W_t = tf.get_variable(
                "W_t",
                [1, int(input_layer.get_shape()[-1]), out_channels],
                tf.float32,
                tf.random_normal_initializer(0, 0.01),
                trainable=True,
            )
            b_t = tf.get_variable(
                "b_t", shape=[out_channels], dtype=tf.float32, initializer=None, trainable=True
            )
            T = tf.nn.bias_add(tf.nn.convolution(input_layer, W_t, "SAME"), b_t)
            T = tf.nn.sigmoid(T)
            residual = H * T + input_layer * (1.0 - T)
        elif in_channels != out_channels:
            W_h = tf.get_variable(
                "W_h",
                [1, int(input_layer.get_shape()[-1]), out_channels],
                tf.float32,
                tf.random_normal_initializer(0, 0.01),
                trainable=True,
            )
            b_h = tf.get_variable(
                "b_h", shape=[out_channels], dtype=tf.float32, initializer=None, trainable=True
            )
            residual = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, "SAME"), b_h)
        else:
            print("no residual convolution")
        res = input_layer if residual is None else residual
        return tf.nn.relu(dropout2 + res)


def temporal_convd(input_layer, num_channels, sequence_length, kernel_size=2, dropout=0):
    for i in range(len(num_channels)):
        dilation_size = 2 ** i
        out_channels = num_channels[i]
        input_layer = temporalblock(
            input_layer, out_channels, kernel_size, 1, dilation_size, dropout, i
        )
        print(input_layer.shape)

    return input_layer


class Model:
    def __init__(
        self,
        embedded_size,
        dict_size,
        dimension_output,
        learning_rate,
        levels=5,
        size_layer=256,
        kernel_size=7,
        maxlen=50,
    ):
        self.X = tf.placeholder(tf.int32, [None, maxlen])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        # channel_sizes = [int(size_layer * ((i+1) / levels)) for i in reversed(range(levels))]
        channel_sizes = [size_layer] * levels
        tcn = temporal_convd(
            input_layer=encoder_embedded,
            num_channels=channel_sizes,
            sequence_length=maxlen,
            kernel_size=kernel_size,
        )
        self.logits = tf.contrib.layers.fully_connected(
            tcn[:, -1, :], dimension_output, activation_fn=None
        )
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[7]:


embedded_size = 128
dimension_output = len(trainset.target_names)
batch_size = 128
learning_rate = 1e-3
maxlen = 50

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(embedded_size, len(dictionary), dimension_output, learning_rate)
sess.run(tf.global_variables_initializer())


# In[8]:


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


# In[9]:


logits = sess.run(model.logits, feed_dict={model.X: str_idx(test_X, dictionary, maxlen)})
print(
    metrics.classification_report(test_Y, np.argmax(logits, 1), target_names=trainset.target_names)
)
