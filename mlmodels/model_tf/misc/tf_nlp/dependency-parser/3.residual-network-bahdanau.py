#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from keras.preprocessing.sequence import pad_sequences

# In[2]:


with open("test.conll.txt") as fopen:
    corpus = fopen.read().split("\n")

with open("dev.conll.txt") as fopen:
    corpus_test = fopen.read().split("\n")


# In[3]:


word2idx = {"PAD": 0, "NUM": 1, "UNK": 2}
tag2idx = {"PAD": 0}
char2idx = {"PAD": 0}
word_idx = 3
tag_idx = 1
char_idx = 1


def process_corpus(corpus, until=None):
    global word2idx, tag2idx, char2idx, word_idx, tag_idx, char_idx
    words, depends, labels = [], [], []
    temp_word, temp_depend, temp_label = [], [], []
    for sentence in corpus:
        if len(sentence):
            sentence = sentence.split("\t")
            for c in sentence[1]:
                if c not in char2idx:
                    char2idx[c] = char_idx
                    char_idx += 1
            if sentence[7] not in tag2idx:
                tag2idx[sentence[7]] = tag_idx
                tag_idx += 1
            if sentence[1] not in word2idx:
                word2idx[sentence[1]] = word_idx
                word_idx += 1
            temp_word.append(word2idx[sentence[1]])
            temp_depend.append(int(sentence[6]))
            temp_label.append(tag2idx[sentence[7]])
        else:
            words.append(temp_word)
            depends.append(temp_depend)
            labels.append(temp_label)
            temp_word = []
            temp_depend = []
            temp_label = []
    return words[:-1], depends[:-1], labels[:-1]


words, depends, labels = process_corpus(corpus)
words_test, depends_test, labels_test = process_corpus(corpus_test)


# In[4]:


# In[5]:


words = pad_sequences(words, padding="post")
depends = pad_sequences(depends, padding="post")
labels = pad_sequences(labels, padding="post")

words_test = pad_sequences(words_test, padding="post")
depends_test = pad_sequences(depends_test, padding="post")
labels_test = pad_sequences(labels_test, padding="post")


# In[6]:


idx2word = {idx: tag for tag, idx in word2idx.items()}
idx2tag = {i: w for w, i in tag2idx.items()}

train_X = words
train_Y = labels
train_depends = depends

test_X = words_test
test_Y = labels_test
test_depends = depends_test


# In[7]:


maxlen = max(train_X.shape[1], test_X.shape[1])

train_X = pad_sequences(train_X, padding="post", maxlen=maxlen)
train_Y = pad_sequences(train_Y, padding="post", maxlen=maxlen)
train_depends = pad_sequences(train_depends, padding="post", maxlen=maxlen)

test_X = pad_sequences(test_X, padding="post", maxlen=maxlen)
test_Y = pad_sequences(test_Y, padding="post", maxlen=maxlen)
test_depends = pad_sequences(test_depends, padding="post", maxlen=maxlen)


# In[8]:


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
    def __init__(self, dict_size, size_layers, learning_rate, maxlen, num_blocks=3, block_size=128):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, maxlen])
        self.labels = tf.placeholder(tf.int32, shape=[None, maxlen])
        self.depends = tf.placeholder(tf.int32, shape=[None, maxlen])
        embeddings = tf.Variable(tf.random_uniform([dict_size, size_layers], -1, 1))
        embedded = tf.nn.embedding_lookup(embeddings, self.word_ids)
        self.attention = Attention(size_layers)
        self.maxlen = tf.shape(self.word_ids)[1]
        self.lengths = tf.count_nonzero(self.word_ids, 1)

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
        logits = tf.layers.conv1d(zeros, len(idx2tag), kernel_size=1, strides=1, padding="SAME")
        logits_depends = tf.layers.conv1d(zeros, maxlen, kernel_size=1, strides=1, padding="SAME")
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            logits, self.labels, self.lengths
        )
        with tf.variable_scope("depends"):
            log_likelihood_depends, transition_params_depends = tf.contrib.crf.crf_log_likelihood(
                logits_depends, self.depends, self.lengths
            )
        self.cost = tf.reduce_mean(-log_likelihood) + tf.reduce_mean(-log_likelihood_depends)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        mask = tf.sequence_mask(self.lengths, maxlen=self.maxlen)

        self.tags_seq, _ = tf.contrib.crf.crf_decode(logits, transition_params, self.lengths)
        self.tags_seq_depends, _ = tf.contrib.crf.crf_decode(
            logits_depends, transition_params_depends, self.lengths
        )

        self.prediction = tf.boolean_mask(self.tags_seq, mask)
        mask_label = tf.boolean_mask(self.labels, mask)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.prediction = tf.boolean_mask(self.tags_seq_depends, mask)
        mask_label = tf.boolean_mask(self.depends, mask)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy_depends = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[9]:


tf.reset_default_graph()
sess = tf.InteractiveSession()

dim = 256
dropout = 1
learning_rate = 1e-3
batch_size = 32

model = Model(len(word2idx), dim, learning_rate, maxlen)
sess.run(tf.global_variables_initializer())


# In[10]:


for e in range(20):
    lasttime = time.time()
    train_acc, train_loss, test_acc, test_loss, train_acc_depends, test_acc_depends = (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    pbar = tqdm(range(0, len(train_X), batch_size), desc="train minibatch loop")
    for i in pbar:
        batch_x = train_X[i : min(i + batch_size, train_X.shape[0])]
        batch_y = train_Y[i : min(i + batch_size, train_X.shape[0])]
        batch_depends = train_depends[i : min(i + batch_size, train_X.shape[0])]
        acc_depends, acc, cost, _ = sess.run(
            [model.accuracy_depends, model.accuracy, model.cost, model.optimizer],
            feed_dict={
                model.word_ids: batch_x,
                model.labels: batch_y,
                model.depends: batch_depends,
            },
        )
        assert not np.isnan(cost)
        train_loss += cost
        train_acc += acc
        train_acc_depends += acc_depends
        pbar.set_postfix(cost=cost, accuracy=acc, accuracy_depends=acc_depends)

    pbar = tqdm(range(0, len(test_X), batch_size), desc="test minibatch loop")
    for i in pbar:
        batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
        batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
        batch_depends = test_depends[i : min(i + batch_size, test_X.shape[0])]
        acc_depends, acc, cost = sess.run(
            [model.accuracy_depends, model.accuracy, model.cost],
            feed_dict={
                model.word_ids: batch_x,
                model.labels: batch_y,
                model.depends: batch_depends,
            },
        )
        assert not np.isnan(cost)
        test_loss += cost
        test_acc += acc
        test_acc_depends += acc_depends
        pbar.set_postfix(cost=cost, accuracy=acc, accuracy_depends=acc_depends)

    train_loss /= len(train_X) / batch_size
    train_acc /= len(train_X) / batch_size
    train_acc_depends /= len(train_X) / batch_size
    test_loss /= len(test_X) / batch_size
    test_acc /= len(test_X) / batch_size
    test_acc_depends /= len(test_X) / batch_size

    print("time taken:", time.time() - lasttime)
    print(
        "epoch: %d, training loss: %f, training acc: %f, training depends: %f, valid loss: %f, valid acc: %f, valid depends: %f\n"
        % (e, train_loss, train_acc, train_acc_depends, test_loss, test_acc, test_acc_depends)
    )


# In[11]:


seq, deps = sess.run(
    [model.tags_seq, model.tags_seq_depends], feed_dict={model.word_ids: batch_x[:1]}
)


# In[12]:


seq = seq[0]
deps = deps[0]


# In[13]:


seq[seq > 0]


# In[14]:


batch_y[0][seq > 0]


# In[15]:


deps[seq > 0]


# In[16]:


batch_depends[0][seq > 0]


# In[ ]:
