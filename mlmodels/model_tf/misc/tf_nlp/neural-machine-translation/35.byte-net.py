#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import os
import re
import time

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

# In[2]:


def build_dataset(words, n_words, atleast=1):
    count = [["PAD", 0], ["GO", 1], ["EOS", 2], ["UNK", 3]]
    counter = collections.Counter(words).most_common(n_words)
    counter = [i for i in counter if i[1] >= atleast]
    count.extend(counter)
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


# In[3]:


with open("english-train", "r") as fopen:
    text_from = fopen.read().lower().split("\n")[:-1]
with open("vietnam-train", "r") as fopen:
    text_to = fopen.read().lower().split("\n")[:-1]
print("len from: %d, len to: %d" % (len(text_from), len(text_to)))


# In[4]:


concat_from = " ".join(text_from).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(
    concat_from, vocabulary_size_from
)
print("vocab from size: %d" % (vocabulary_size_from))
print("Most common words", count_from[4:10])
print("Sample data", data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])


# In[5]:


concat_to = " ".join(text_to).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, vocabulary_size_to)
print("vocab to size: %d" % (vocabulary_size_to))
print("Most common words", count_to[4:10])
print("Sample data", data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])


# In[6]:


GO = dictionary_from["GO"]
PAD = dictionary_from["PAD"]
EOS = dictionary_from["EOS"]
UNK = dictionary_from["UNK"]


# In[7]:


for i in range(len(text_to)):
    text_to[i] += " EOS"


# In[8]:


def str_idx(corpus, dic):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k, UNK))
        X.append(ints)
    return X


def pad_sentence_batch(sentence_batch, pad_int, maxlen):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = maxlen
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(maxlen)
    return padded_seqs, seq_lens


# In[9]:


X = str_idx(text_from, dictionary_from)
Y = str_idx(text_to, dictionary_to)


# In[10]:


maxlen_question = max([len(x) for x in X]) * 2
maxlen_answer = max([len(y) for y in Y]) * 2


# In[11]:


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
        print(dilated_conv)
        relu3 = tf.nn.relu(layer_normalization(dilated_conv))
        conv2 = conv1d(relu3, 2 * residual_channels)
        return input_ + conv2


class ByteNet:
    def __init__(
        self,
        from_vocab_size,
        to_vocab_size,
        channels,
        encoder_dilations,
        decoder_dilations,
        encoder_filter_width,
        decoder_filter_width,
        learning_rate=0.001,
        beta1=0.5,
    ):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        target_1 = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        embedding_channels = 2 * channels
        w_source_embedding = tf.Variable(
            tf.random_normal([from_vocab_size, embedding_channels], stddev=0.02)
        )
        w_target_embedding = tf.Variable(
            tf.random_normal([to_vocab_size, embedding_channels], stddev=0.02)
        )
        source_embedding = tf.nn.embedding_lookup(w_source_embedding, self.X)
        target_1_embedding = tf.nn.embedding_lookup(w_target_embedding, target_1)
        curr_input = source_embedding
        for layer_no, dilation in enumerate(encoder_dilations):
            curr_input = bytenet_residual_block(
                curr_input, dilation, layer_no, channels, encoder_filter_width, causal=False
            )
        encoder_output = curr_input
        combined_embedding = target_1_embedding + encoder_output
        curr_input = combined_embedding
        for layer_no, dilation in enumerate(decoder_dilations):
            curr_input = bytenet_residual_block(
                curr_input, dilation, layer_no, channels, encoder_filter_width, causal=False
            )
        self.logits = conv1d(curr_input, to_vocab_size)
        masks = tf.sequence_mask(self.Y_seq_len, maxlen_answer, dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(
            logits=self.logits, targets=self.Y, weights=masks
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.logits, axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[12]:


residual_channels = 128
encoder_dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
decoder_dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
encoder_filter_width = 3
decoder_filter_width = 3
batch_size = 16
epoch = 20


# In[13]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = ByteNet(
    len(dictionary_from),
    len(dictionary_to),
    residual_channels,
    encoder_dilations,
    decoder_dilations,
    encoder_filter_width,
    decoder_filter_width,
)
sess.run(tf.global_variables_initializer())


# In[14]:


for i in range(epoch):
    total_loss, total_accuracy = 0, 0
    X, Y = shuffle(X, Y)
    for k in range(0, len(text_to), batch_size):
        index = min(k + batch_size, len(text_to))
        batch_x, seq_x = pad_sentence_batch(X[k:index], PAD, maxlen_answer)
        batch_y, seq_y = pad_sentence_batch(Y[k:index], PAD, maxlen_answer)
        predicted, accuracy, loss, _ = sess.run(
            [tf.argmax(model.logits, 2), model.accuracy, model.cost, model.optimizer],
            feed_dict={
                model.X: batch_x,
                model.Y: batch_y,
                model.X_seq_len: seq_x,
                model.Y_seq_len: seq_y,
            },
        )
        total_loss += loss
        total_accuracy += accuracy
    total_loss /= len(text_to) / batch_size
    total_accuracy /= len(text_to) / batch_size
    print("epoch: %d, avg loss: %f, avg accuracy: %f" % (i + 1, total_loss, total_accuracy))


# In[16]:


for i in range(len(batch_x)):
    print("row %d" % (i + 1))
    print(
        "QUESTION:", " ".join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0, 1, 2, 3]])
    )
    print(
        "REAL ANSWER:",
        " ".join([rev_dictionary_to[n] for n in batch_y[i] if n not in [0, 1, 2, 3]]),
    )
    print(
        "PREDICTED ANSWER:",
        " ".join([rev_dictionary_to[n] for n in predicted[i] if n not in [0, 1, 2, 3]]),
        "\n",
    )


# In[ ]:
