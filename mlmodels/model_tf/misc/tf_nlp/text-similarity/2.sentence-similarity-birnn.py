#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import random
import time

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from tqdm import tqdm

# In[2]:


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
        for no, k in enumerate(corpus[i][:maxlen][::-1]):
            val = dic[k] if k in dic else UNK
            X[i, -1 - no] = val
    return X


def load_data(filepath):
    x1 = []
    x2 = []
    y = []
    for line in open(filepath):
        l = line.strip().split("\t")
        if len(l) < 2:
            continue
        if random.random() > 0.5:
            x1.append(l[0].lower())
            x2.append(l[1].lower())
        else:
            x1.append(l[1].lower())
            x2.append(l[0].lower())
        y.append(int(l[2]))
    return np.array(x1), np.array(x2), np.array(y)


# In[3]:


X1_text, X2_text, Y = load_data("train_snli.txt")


# In[4]:


np.unique(Y, return_counts=True)


# In[5]:


concat = (" ".join(X1_text.tolist() + X2_text.tolist())).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print("vocab from size: %d" % (vocabulary_size))
print("Most common words", count[4:10])
print("Sample data", data[:10], [rev_dictionary[i] for i in data[:10]])


# In[6]:


class Model:
    def __init__(self, size_layer, num_layers, embedded_size, dict_size, learning_rate, dropout):
        def cells(size, reuse=False):
            cell = tf.nn.rnn_cell.LSTMCell(
                size, initializer=tf.orthogonal_initializer(), reuse=reuse
            )
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)

        def birnn(inputs, scope):
            with tf.variable_scope(scope):
                for n in range(num_layers):
                    (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=cells(size_layer // 2),
                        cell_bw=cells(size_layer // 2),
                        inputs=inputs,
                        dtype=tf.float32,
                        scope="bidirectional_rnn_%d" % (n),
                    )
                    inputs = tf.concat((out_fw, out_bw), 2)
                return inputs[:, -1]

        self.X_left = tf.placeholder(tf.int32, [None, None])
        self.X_right = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None])
        self.batch_size = tf.shape(self.X_left)[0]
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        embedded_left = tf.nn.embedding_lookup(encoder_embeddings, self.X_left)
        embedded_right = tf.nn.embedding_lookup(encoder_embeddings, self.X_right)

        def contrastive_loss(y, d):
            tmp = y * tf.square(d)
            tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
            return tf.reduce_sum(tmp + tmp2) / tf.cast(self.batch_size, tf.float32) / 2

        self.output_left = birnn(embedded_left, "left")
        self.output_right = birnn(embedded_right, "right")
        self.distance = tf.sqrt(
            tf.reduce_sum(
                tf.square(tf.subtract(self.output_left, self.output_right)), 1, keep_dims=True
            )
        )
        self.distance = tf.div(
            self.distance,
            tf.add(
                tf.sqrt(tf.reduce_sum(tf.square(self.output_left), 1, keep_dims=True)),
                tf.sqrt(tf.reduce_sum(tf.square(self.output_right), 1, keep_dims=True)),
            ),
        )
        self.distance = tf.reshape(self.distance, [-1])
        self.cost = contrastive_loss(self.Y, self.distance)

        self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance))
        correct_predictions = tf.equal(self.temp_sim, self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


# In[7]:


size_layer = 256
num_layers = 2
embedded_size = 128
learning_rate = 1e-3
maxlen = 50
batch_size = 128
dropout = 0.8


# In[8]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(size_layer, num_layers, embedded_size, len(dictionary), learning_rate, dropout)
sess.run(tf.global_variables_initializer())


# In[9]:


vectors_left = str_idx(X1_text, dictionary, maxlen)
vectors_right = str_idx(X2_text, dictionary, maxlen)
train_X_left, test_X_left, train_X_right, test_X_right, train_Y, test_Y = train_test_split(
    vectors_left, vectors_right, Y, test_size=0.2
)


# In[10]:


for EPOCH in range(5):
    lasttime = time.time()

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    pbar = tqdm(range(0, len(train_X_left), batch_size), desc="train minibatch loop")
    for i in pbar:
        batch_x_left = train_X_left[i : min(i + batch_size, train_X_left.shape[0])]
        batch_x_right = train_X_right[i : min(i + batch_size, train_X_left.shape[0])]
        batch_y = train_Y[i : min(i + batch_size, train_X_left.shape[0])]
        acc, loss, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X_left: batch_x_left, model.X_right: batch_x_right, model.Y: batch_y},
        )
        assert not np.isnan(loss)
        train_loss += loss
        train_acc += acc
        pbar.set_postfix(cost=loss, accuracy=acc)

    pbar = tqdm(range(0, len(test_X_left), batch_size), desc="test minibatch loop")
    for i in pbar:
        batch_x_left = test_X_left[i : min(i + batch_size, train_X_left.shape[0])]
        batch_x_right = test_X_right[i : min(i + batch_size, train_X_left.shape[0])]
        batch_y = test_Y[i : min(i + batch_size, train_X_left.shape[0])]
        acc, loss = sess.run(
            [model.accuracy, model.cost],
            feed_dict={model.X_left: batch_x_left, model.X_right: batch_x_right, model.Y: batch_y},
        )
        test_loss += loss
        test_acc += acc
        pbar.set_postfix(cost=loss, accuracy=acc)

    train_loss /= len(train_X_left) / batch_size
    train_acc /= len(train_X_left) / batch_size
    test_loss /= len(test_X_left) / batch_size
    test_acc /= len(test_X_left) / batch_size

    print("time taken:", time.time() - lasttime)
    print(
        "epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n"
        % (EPOCH, train_loss, train_acc, test_loss, test_acc)
    )


# In[15]:


left = str_idx(["a person is outdoors, on a horse."], dictionary, maxlen)
right = str_idx(["a person on a horse jumps over a broken down airplane."], dictionary, maxlen)
sess.run([model.temp_sim, 1 - model.distance], feed_dict={model.X_left: left, model.X_right: right})


# In[17]:


left = str_idx(["i love you"], dictionary, maxlen)
right = str_idx(["you love i"], dictionary, maxlen)
sess.run([model.temp_sim, 1 - model.distance], feed_dict={model.X_left: left, model.X_right: right})


# In[ ]:
