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


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


# In[9]:


X = str_idx(text_from, dictionary_from)
Y = str_idx(text_to, dictionary_to)


# In[10]:


emb_size = 256
n_hidden = 256
n_layers = 4
n_attn_heads = 16
learning_rate = 1e-4
batch_size = 16
epoch = 20


# In[11]:


def encoder_block(inp, n_hidden, filter_size):
    inp = tf.expand_dims(inp, 2)
    inp = tf.pad(
        inp, [[0, 0], [(filter_size[0] - 1) // 2, (filter_size[0] - 1) // 2], [0, 0], [0, 0]]
    )
    conv = tf.layers.conv2d(inp, n_hidden, filter_size, padding="VALID", activation=None)
    conv = tf.squeeze(conv, 2)
    return conv


def decoder_block(inp, n_hidden, filter_size):
    inp = tf.expand_dims(inp, 2)
    inp = tf.pad(inp, [[0, 0], [filter_size[0] - 1, 0], [0, 0], [0, 0]])
    conv = tf.layers.conv2d(inp, n_hidden, filter_size, padding="VALID", activation=None)
    conv = tf.squeeze(conv, 2)
    return conv


def glu(x):
    return tf.multiply(x[:, :, : tf.shape(x)[2] // 2], tf.sigmoid(x[:, :, tf.shape(x)[2] // 2 :]))


def layer(inp, conv_block, kernel_width, n_hidden, residual=None):
    z = conv_block(inp, n_hidden, (kernel_width, 1))
    return glu(z) + (residual if residual is not None else 0)


# In[12]:


class Chatbot:
    def __init__(self):

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])

        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)

        encoder_embedding = tf.Variable(tf.random_uniform([len(dictionary_from), emb_size], -1, 1))
        decoder_embedding = tf.Variable(tf.random_uniform([len(dictionary_to), emb_size], -1, 1))

        def forward(x, y, reuse=False):
            with tf.variable_scope("forward", reuse=reuse):
                encoder_embedded = tf.nn.embedding_lookup(encoder_embedding, x)
                decoder_embedded = tf.nn.embedding_lookup(decoder_embedding, y)

                e = tf.identity(encoder_embedded)

                for i in range(n_layers):
                    z = layer(encoder_embedded, encoder_block, 3, n_hidden * 2, encoder_embedded)
                    encoder_embedded = z

                encoder_output, output_memory = z, z + e
                g = tf.identity(decoder_embedded)

                for i in range(n_layers):
                    attn_res = h = layer(
                        decoder_embedded,
                        decoder_block,
                        3,
                        n_hidden * 2,
                        residual=tf.zeros_like(decoder_embedded),
                    )
                    C = []
                    for j in range(n_attn_heads):
                        h_ = tf.layers.dense(h, n_hidden // n_attn_heads)
                        g_ = tf.layers.dense(g, n_hidden // n_attn_heads)
                        zu_ = tf.layers.dense(encoder_output, n_hidden // n_attn_heads)
                        ze_ = tf.layers.dense(output_memory, n_hidden // n_attn_heads)

                        d = tf.layers.dense(h_, n_hidden // n_attn_heads) + g_
                        dz = tf.matmul(d, tf.transpose(zu_, [0, 2, 1]))
                        a = tf.nn.softmax(dz)
                        c_ = tf.matmul(a, ze_)
                        C.append(c_)

                    c = tf.concat(C, 2)
                    h = tf.layers.dense(attn_res + c, n_hidden)
                    decoder_embedded = h

                decoder_output = tf.sigmoid(h)
                return tf.layers.dense(decoder_output, len(dictionary_to))

        self.training_logits = forward(self.X, decoder_input)
        self.logits = forward(self.X, self.Y, reuse=True)
        self.k = tf.placeholder(dtype=tf.int32)
        p = tf.nn.softmax(self.logits)
        self.topk_logprobs, self.topk_ids = tf.nn.top_k(tf.log(p), self.k)

        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(
            logits=self.training_logits, targets=self.Y, weights=masks
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.training_logits, axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[13]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Chatbot()
sess.run(tf.global_variables_initializer())


# In[14]:


for i in range(epoch):
    total_loss, total_accuracy = 0, 0
    for k in range(0, len(text_to), batch_size):
        index = min(k + batch_size, len(text_to))
        batch_x, seq_x = pad_sentence_batch(X[k:index], PAD)
        batch_y, seq_y = pad_sentence_batch(Y[k:index], PAD)
        accuracy, loss, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X: batch_x, model.Y: batch_y},
        )
        total_loss += loss
        total_accuracy += accuracy
    total_loss /= len(text_to) / batch_size
    total_accuracy /= len(text_to) / batch_size
    print("epoch: %d, avg loss: %f, avg accuracy: %f" % (i + 1, total_loss, total_accuracy))


# In[15]:


class Hypothesis:
    def __init__(self, log_prob, seq):
        self.log_prob = log_prob
        self.seq = seq

    @property
    def step(self):
        return len(self.seq) - 1


def beam_search(batch_x, beam_size, num_ans=5, normalize_by_len=1.0):
    assert 0 <= normalize_by_len <= 1
    batch_size = len(batch_x)
    max_len = len(batch_x[0]) * 2
    dec_inputs = np.ones((batch_size, 2), dtype=np.int32)
    answers = [[] for i in range(batch_size)]
    H = [[] for i in range(batch_size)]

    tkl, tkid = sess.run(
        [model.topk_logprobs, model.topk_ids],
        feed_dict={model.X: batch_x, model.Y: dec_inputs, model.k: beam_size},
    )
    for i in range(batch_size):
        for j, log_prob in enumerate(tkl[i, 0]):
            if tkid[i, 0, j] != EOS:
                h = Hypothesis(log_prob, [1, tkid[i, 0, j]])
                H[i].append(h)
        H[i].sort(key=lambda h: h.log_prob)

    done = [False] * batch_size
    while not all(done):
        tkl_beam = []
        tkid_beam = []
        dec_inputs_beam = []
        steps_beam = []
        for i in range(beam_size):
            steps = [1] * batch_size
            prev_log_probs = np.zeros(batch_size, dtype=np.float32)
            dec_inputs = np.ones((batch_size, max_len), dtype=np.int32)
            for j, h in enumerate(H):
                while h:
                    hi = h.pop()
                    lp, step, candidate_seq = hi.log_prob, hi.step, hi.seq
                    if candidate_seq[-1] != EOS:
                        dec_inputs[j, : len(candidate_seq)] = candidate_seq
                        steps[j] = step
                        prev_log_probs[j] = lp
                        break
                    else:
                        answers[j].append((lp, candidate_seq))
            max_step = max(steps)
            dec_inputs = dec_inputs[:, : max_step + 2]
            tkl, tkid = sess.run(
                [model.topk_logprobs, model.topk_ids],
                feed_dict={model.X: batch_x, model.Y: dec_inputs, model.k: beam_size},
            )

            tkl_beam.append(tkl + prev_log_probs[:, None, None])
            tkid_beam.append(tkid)
            dec_inputs_beam.append(dec_inputs.copy())
            steps_beam.append(steps)

        for i in range(beam_size):
            tkl = tkl_beam[i]
            tkid = tkid_beam[i]
            dec_inputs = dec_inputs_beam[i]
            steps = steps_beam[i]
            for j in range(batch_size):
                step = steps[j]
                for k in range(tkid.shape[2]):
                    extended_seq = np.hstack((dec_inputs[j, : step + 1], [tkid[j, step, k]]))
                    log_prob = tkl[j, step, k]
                    if len(extended_seq) <= max_len and log_prob > -10:
                        h = Hypothesis(log_prob, extended_seq)
                        H[j].append(h)
                H[j].sort(key=lambda h: h.log_prob / (h.step ** normalize_by_len))

        for i in range(batch_size):
            done[i] = (len(answers[i]) >= num_ans) or (not H[i]) or (len(H[i]) > 100)

    return answers


# In[16]:


beamed = beam_search(batch_x, 5)


# In[17]:


beamed = [i for i in beamed if len(i)]


# In[18]:


predicted = [max(b, key=lambda t: t[0])[1] for b in beamed]


# In[23]:


for i in range(len(predicted)):
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
