#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
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


def contruct_cells(hidden_structs):
    cells = []
    for hidden_dims in hidden_structs:
        cells.append(tf.contrib.rnn.LSTMCell(hidden_dims))
    return cells


def rnn_reformat(x, input_dims, n_steps):
    x_ = tf.transpose(x, [1, 0, 2])
    x_ = tf.reshape(x_, [-1, input_dims])
    return tf.split(x_, n_steps, 0)


def dilated_rnn(cell, inputs, rate, scope="default"):
    n_steps = len(inputs)
    if not (n_steps % rate) == 0:
        zero_tensor = tf.zeros_like(inputs[0])
        dilated_n_steps = n_steps // rate + 1
        for i_pad in range(dilated_n_steps * rate - n_steps):
            inputs.append(zero_tensor)
    else:
        dilated_n_steps = n_steps // rate
    dilated_inputs = [
        tf.concat(inputs[i * rate : (i + 1) * rate], axis=0) for i in range(dilated_n_steps)
    ]
    dilated_outputs, _ = tf.contrib.rnn.static_rnn(
        cell, dilated_inputs, dtype=tf.float32, scope=scope
    )
    splitted_outputs = [tf.split(output, rate, axis=0) for output in dilated_outputs]
    unrolled_outputs = [output for sublist in splitted_outputs for output in sublist]
    return unrolled_outputs[:n_steps]


def multi_dilated_rnn(cells, inputs, dilations):
    x = copy.copy(inputs)
    for cell, dilation in zip(cells, dilations):
        x = dilated_rnn(cell, x, dilation, scope="multi_dilated_rnn_%d" % dilation)
    return x


class Model:
    def __init__(
        self,
        steps,
        dict_size,
        dimension_input,
        dimension_output,
        learning_rate=1e-2,
        hidden_structs=[20],
        dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    ):
        hidden_structs = hidden_structs * len(dilations)
        self.X = tf.placeholder(tf.int32, [None, steps])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, dimension_input], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        x_reformat = rnn_reformat(encoder_embedded, dimension_input, steps)
        cells = contruct_cells(hidden_structs)
        layer_outputs = multi_dilated_rnn(cells, x_reformat, dilations)
        if dilations[0] == 1:
            weights = tf.Variable(tf.random_normal(shape=[hidden_structs[-1], dimension_output]))
            bias = tf.Variable(tf.random_normal(shape=[dimension_output]))
            self.logits = tf.matmul(layer_outputs[-1], weights) + bias
        else:
            weights = tf.Variable(
                tf.random_normal(shape=[hidden_structs[-1] * dilations[0], dimension_output])
            )
            bias = tf.Variable(tf.random_normal(shape=[dimension_output]))
            for idx, i in enumerate(range(-dilations[0], 0, 1)):
                if idx == 0:
                    hidden_outputs_ = layer_outputs[i]
                else:
                    hidden_outputs_ = tf.concat([hidden_outputs_, layer_outputs[i]], axis=1)
            self.logits = tf.matmul(hidden_outputs_, weights) + bias
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


# In[7]:


embedded_size = 128
dimension_output = len(trainset.target_names)
maxlen = 50
batch_size = 128


# In[8]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(maxlen, vocabulary_size + 4, embedded_size, dimension_output)
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


# In[10]:


logits = sess.run(model.logits, feed_dict={model.X: str_idx(test_X, dictionary, maxlen)})
print(
    metrics.classification_report(test_Y, np.argmax(logits, 1), target_names=trainset.target_names)
)


# In[ ]:
