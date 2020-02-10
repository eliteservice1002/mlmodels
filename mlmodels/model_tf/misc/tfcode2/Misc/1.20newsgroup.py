#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import re
import time

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups

# In[2]:


newsgroups_train = fetch_20newsgroups(subset="train")
newsgroups_test = fetch_20newsgroups(subset="test")


# In[3]:


def clearstring(string):
    string = re.sub("[^A-Za-z ]+", "", string)
    string = string.split("\n")
    string = [y.strip() for y in filter(None, string)]
    string = (" ".join(string)).lower()
    return " ".join([y.strip() for y in string.split()])


def build_dataset(words, n_words):
    count = [["GO", 0], ["PAD", 1], ["EOS", 2], ["UNK", 3]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def str_idx(corpus, dic, maxlen, UNK=3):
    X = np.zeros((len(corpus), maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
            try:
                X[i, -1 - no] = dic[k]
            except Exception as e:
                X[i, -1 - no] = UNK
    return X


# In[4]:


for i in range(len(newsgroups_train.data)):
    newsgroups_train.data[i] = clearstring(newsgroups_train.data[i])

for i in range(len(newsgroups_test.data)):
    newsgroups_test.data[i] = clearstring(newsgroups_test.data[i])


# In[5]:


concat = " ".join(newsgroups_train.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print("vocab from size: %d" % (vocabulary_size))
print("Most common words", count[4:10])
print("Sample data", data[:10], [rev_dictionary[i] for i in data[:10]])


# In[6]:


GO = dictionary["GO"]
PAD = dictionary["PAD"]
EOS = dictionary["EOS"]
UNK = dictionary["UNK"]


# In[7]:


class Model:
    def __init__(
        self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate
    ):
        def cells(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(
                size_layer, initializer=tf.orthogonal_initializer(), reuse=reuse
            )

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None])
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.name_scope("layer_embedded"):
            encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
            encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)

            tf.summary.histogram("X", self.X)
            tf.summary.histogram("Embedded", encoder_embeddings)

        with tf.name_scope("layer_rnn"):
            rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
            outputs, _ = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded, dtype=tf.float32)

        with tf.name_scope("layer_logits"):
            W = tf.get_variable(
                "w", shape=(size_layer, dimension_output), initializer=tf.orthogonal_initializer()
            )
            b = tf.get_variable("b", shape=(dimension_output), initializer=tf.zeros_initializer())
            self.logits = tf.matmul(outputs[:, -1], W) + b

            tf.summary.histogram("Weight", W)
            tf.summary.histogram("logits", self.logits)

        with tf.name_scope("optimizer"):
            self.cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
            )
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                self.cost, global_step=self.global_step
            )
            tf.summary.scalar("cost", self.cost)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.cast(self.Y, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)


# In[8]:


size_layer = 128
num_layers = 2
embedded_size = 128
dimension_output = len(newsgroups_train.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 128


# In[9]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(
    size_layer, num_layers, embedded_size, vocabulary_size + 4, dimension_output, learning_rate
)
sess.run(tf.global_variables_initializer())


# In[10]:


merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs", sess.graph)


# In[11]:


train_X = newsgroups_train.data
train_Y = newsgroups_train.target
test_X = newsgroups_test.data
test_Y = newsgroups_test.target


# In[ ]:


EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 2, 0, 0, 0
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
            feed_dict={model.X: batch_x, model.Y: train_Y[i : i + batch_size]},
        )
        train_loss += loss
        train_acc += acc
        summary = sess.run(
            merged, feed_dict={model.X: batch_x, model.Y: train_Y[i : i + batch_size]}
        )
        writer.add_summary(summary, global_step=sess.run(model.global_step))

    for i in range(0, (len(test_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(test_X[i : i + batch_size], dictionary, maxlen)
        acc, loss = sess.run(
            [model.accuracy, model.cost],
            feed_dict={model.X: batch_x, model.Y: test_Y[i : i + batch_size]},
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


get_ipython().system("tensorboard --logdir=./logs")
