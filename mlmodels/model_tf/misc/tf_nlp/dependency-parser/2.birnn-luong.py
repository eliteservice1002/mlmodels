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
char2idx = {"PAD": 0, "NUM": 1, "UNK": 2}
word_idx = 3
tag_idx = 1
char_idx = 3


def process_corpus(corpus, until=None):
    global word2idx, tag2idx, char2idx, word_idx, tag_idx, char_idx
    sentences, words, depends, labels = [], [], [], []
    temp_sentence, temp_word, temp_depend, temp_label = [], [], [], []
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
            temp_sentence.append(sentence[1])
        else:
            words.append(temp_word)
            depends.append(temp_depend)
            labels.append(temp_label)
            sentences.append(temp_sentence)
            temp_word = []
            temp_depend = []
            temp_label = []
            temp_sentence = []
    return sentences[:-1], words[:-1], depends[:-1], labels[:-1]


sentences, words, depends, labels = process_corpus(corpus)
sentences_test, words_test, depends_test, labels_test = process_corpus(corpus_test)


# In[4]:


# In[5]:


words = pad_sequences(words, padding="post")
depends = pad_sequences(depends, padding="post")
labels = pad_sequences(labels, padding="post")

words_test = pad_sequences(words_test, padding="post")
depends_test = pad_sequences(depends_test, padding="post")
labels_test = pad_sequences(labels_test, padding="post")


# In[6]:


words_test.shape


# In[7]:


def generate_char_seq(batch, UNK=2):
    maxlen_c = max([len(k) for k in batch])
    x = [[len(i) for i in k] for k in batch]
    maxlen = max([j for i in x for j in i])
    temp = np.zeros((len(batch), maxlen_c, maxlen), dtype=np.int32)
    for i in range(len(batch)):
        for k in range(len(batch[i])):
            for no, c in enumerate(batch[i][k]):
                temp[i, k, -1 - no] = char2idx.get(c, UNK)
    return temp


# In[8]:


idx2word = {idx: tag for tag, idx in word2idx.items()}
idx2tag = {i: w for w, i in tag2idx.items()}

train_X = words
train_Y = labels
train_depends = depends
train_char = generate_char_seq(sentences)

test_X = words_test
test_Y = labels_test
test_depends = depends_test
test_char = generate_char_seq(sentences_test)


# In[9]:


class Model:
    def __init__(
        self,
        dim_word,
        dim_char,
        dropout,
        learning_rate,
        hidden_size_char,
        hidden_size_word,
        num_layers,
        maxlen,
    ):
        def cells(size, reuse=False):
            return tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(), reuse=reuse),
                output_keep_prob=dropout,
            )

        def bahdanau(embedded, size):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=hidden_size_word, memory=embedded
            )
            return tf.contrib.seq2seq.AttentionWrapper(
                cell=cells(hidden_size_word),
                attention_mechanism=attention_mechanism,
                attention_layer_size=hidden_size_word,
            )

        self.word_ids = tf.placeholder(tf.int32, shape=[None, None])
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None])
        self.labels = tf.placeholder(tf.int32, shape=[None, None])
        self.depends = tf.placeholder(tf.int32, shape=[None, None])
        self.maxlen = tf.shape(self.word_ids)[1]
        self.lengths = tf.count_nonzero(self.word_ids, 1)

        self.word_embeddings = tf.Variable(
            tf.truncated_normal([len(word2idx), dim_word], stddev=1.0 / np.sqrt(dim_word))
        )
        self.char_embeddings = tf.Variable(
            tf.truncated_normal([len(char2idx), dim_char], stddev=1.0 / np.sqrt(dim_char))
        )

        word_embedded = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids)
        char_embedded = tf.nn.embedding_lookup(self.char_embeddings, self.char_ids)
        s = tf.shape(char_embedded)
        char_embedded = tf.reshape(char_embedded, shape=[s[0] * s[1], s[-2], dim_char])

        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cells(hidden_size_char),
                cell_bw=cells(hidden_size_char),
                inputs=char_embedded,
                dtype=tf.float32,
                scope="bidirectional_rnn_char_%d" % (n),
            )
            char_embedded = tf.concat((out_fw, out_bw), 2)
        output = tf.reshape(char_embedded[:, -1], shape=[s[0], s[1], 2 * hidden_size_char])
        word_embedded = tf.concat([word_embedded, output], axis=-1)

        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=bahdanau(word_embedded, hidden_size_word),
                cell_bw=bahdanau(word_embedded, hidden_size_word),
                inputs=word_embedded,
                dtype=tf.float32,
                scope="bidirectional_rnn_word_%d" % (n),
            )
            word_embedded = tf.concat((out_fw, out_bw), 2)

        logits = tf.layers.dense(word_embedded, len(idx2tag))
        logits_depends = tf.layers.dense(word_embedded, maxlen)
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


# In[10]:


tf.reset_default_graph()
sess = tf.InteractiveSession()

dim_word = 128
dim_char = 256
dropout = 1
learning_rate = 1e-3
hidden_size_char = 64
hidden_size_word = 64
num_layers = 2
batch_size = 32

model = Model(
    dim_word,
    dim_char,
    dropout,
    learning_rate,
    hidden_size_char,
    hidden_size_word,
    num_layers,
    words.shape[1],
)
sess.run(tf.global_variables_initializer())


# In[11]:


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
        batch_char = train_char[i : min(i + batch_size, train_X.shape[0])]
        batch_y = train_Y[i : min(i + batch_size, train_X.shape[0])]
        batch_depends = train_depends[i : min(i + batch_size, train_X.shape[0])]
        acc_depends, acc, cost, _ = sess.run(
            [model.accuracy_depends, model.accuracy, model.cost, model.optimizer],
            feed_dict={
                model.word_ids: batch_x,
                model.char_ids: batch_char,
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
        batch_char = test_char[i : min(i + batch_size, test_X.shape[0])]
        batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
        batch_depends = test_depends[i : min(i + batch_size, test_X.shape[0])]
        acc_depends, acc, cost = sess.run(
            [model.accuracy_depends, model.accuracy, model.cost],
            feed_dict={
                model.word_ids: batch_x,
                model.char_ids: batch_char,
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


# In[12]:


seq, deps = sess.run(
    [model.tags_seq, model.tags_seq_depends],
    feed_dict={model.word_ids: batch_x[:1], model.char_ids: batch_char[:1]},
)


# In[13]:


seq = seq[0]
deps = deps[0]


# In[14]:


seq[seq > 0]


# In[15]:


batch_y[0][seq > 0]


# In[16]:


deps[seq > 0]


# In[17]:


batch_depends[0][seq > 0]


# In[ ]:
