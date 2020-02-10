#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import json
import re
import time

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from tqdm import tqdm

from unidecode import unidecode

# In[2]:


with open("ctexts.json", "r") as fopen:
    ctexts = json.load(fopen)

with open("headlines.json", "r") as fopen:
    headlines = json.load(fopen)


# In[3]:


def topic_modelling(string, n=500):
    vectorizer = TfidfVectorizer()
    tf = vectorizer.fit_transform([string])
    tf_features = vectorizer.get_feature_names()
    compose = TruncatedSVD(1).fit(tf)
    return " ".join([tf_features[i] for i in compose.components_[0].argsort()[: -n - 1 : -1]])


# In[4]:


def textcleaning(string):
    string = unidecode(string).replace(".", ". ").replace(",", " , ")
    string = re.sub("[^'\"A-Za-z\- ]+", " ", string)
    string = re.sub(r"[ ]+", " ", string.lower()).strip()
    return string


# In[5]:


get_ipython().run_cell_magic(
    "time",
    "",
    "h, c = [], []\nfor i in range(len(ctexts)):\n    try:\n        c.append(textcleaning(topic_modelling(ctexts[i])))\n        h.append(textcleaning(headlines[i]))\n    except:\n        pass",
)


# In[6]:


def build_dataset(words, n_words):
    count = [["PAD", 0], ["GO", 1], ["EOS", 2], ["UNK", 3]]
    count.extend(collections.Counter(words).most_common(n_words))
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


# In[7]:


concat_from = " ".join(c).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(
    concat_from, vocabulary_size_from
)
print("vocab from size: %d" % (vocabulary_size_from))
print("Most common words", count_from[4:10])
print("Sample data", data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])


# In[8]:


concat_to = " ".join(h).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, vocabulary_size_to)
print("vocab to size: %d" % (vocabulary_size_to))
print("Most common words", count_to[4:10])
print("Sample data", data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])


# In[9]:


for i in range(len(h)):
    h[i] = h[i] + " EOS"
h[0]


# In[10]:


GO = dictionary_from["GO"]
PAD = dictionary_from["PAD"]
EOS = dictionary_from["EOS"]
UNK = dictionary_from["UNK"]


# In[11]:


def str_idx(corpus, dic, UNK=3):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k, UNK))
        X.append(ints)
    return X


# In[12]:


X = str_idx(c, dictionary_from)
Y = str_idx(h, dictionary_to)


# In[13]:


train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)


# In[14]:


class Summarization:
    def __init__(
        self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size
    ):
        def lstm_cell(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(
                size_layer, initializer=tf.orthogonal_initializer(), reuse=reuse
            )

        def attention(encoder_out, seq_len, reuse=False):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=size_layer, memory=encoder_out, memory_sequence_length=seq_len
            )
            return tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell(reuse) for _ in range(num_layers)]),
                attention_mechanism=attention_mechanism,
                attention_layer_size=size_layer,
            )

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]

        # encoder
        encoder_embeddings = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        encoder_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
        self.encoder_out, self.encoder_state = tf.nn.dynamic_rnn(
            cell=encoder_cells,
            inputs=encoder_embedded,
            sequence_length=self.X_seq_len,
            dtype=tf.float32,
        )
        self.encoder_state2 = self.encoder_state

        self.encoder_state = tuple(self.encoder_state[-1] for _ in range(num_layers))
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        # decoder
        decoder_embeddings = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))
        decoder_cell = attention(self.encoder_out, self.X_seq_len)
        self.decoder_cell = decoder_cell
        dense_layer = tf.layers.Dense(to_dict_size)
        decoded = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=decoded, sequence_length=self.Y_seq_len, time_major=False
        )
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=training_helper,
            initial_state=decoder_cell.zero_state(batch_size, tf.float32).clone(
                cell_state=self.encoder_state
            ),
            output_layer=dense_layer,
        )
        training_decoder_output, self.final_state, self.final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder=training_decoder,
            impute_finished=True,
            maximum_iterations=tf.reduce_max(self.Y_seq_len),
        )
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=encoder_embeddings,
            start_tokens=tf.tile(tf.constant([GO], dtype=tf.int32), [batch_size]),
            end_token=EOS,
        )
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=predicting_helper,
            initial_state=decoder_cell.zero_state(batch_size, tf.float32).clone(
                cell_state=self.encoder_state
            ),
            output_layer=dense_layer,
        )
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=predicting_decoder,
            impute_finished=True,
            maximum_iterations=tf.reduce_max(self.X_seq_len),
        )

        context_vector = self.final_state.attention
        state_tuple = self.final_state.cell_state[-1]
        p_gen = context_vector * state_tuple.c * state_tuple.h * decoded[:, -1]
        p_gen = tf.layers.dense(p_gen, 1, activation=tf.nn.sigmoid)
        align = self.final_state.alignments

        vocab_dists = training_decoder_output.rnn_output[:, -1] * p_gen
        attn_dists = (1 - p_gen) * align
        extended_vsize = to_dict_size + 1
        extra_zeros = tf.fill([batch_size, 1], 0.0)
        vocab_dists_extended = tf.concat(axis=1, values=[vocab_dists, extra_zeros])
        batch_nums = tf.range(0, limit=batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)
        attn_len = tf.shape(self.X)[1]
        batch_nums = tf.tile(batch_nums, [1, attn_len])
        indices = tf.stack((batch_nums, self.X), axis=2)
        shape = [batch_size, extended_vsize]
        attn_dists_projected = tf.scatter_nd(indices, attn_dists, shape)
        final_dists = tf.nn.relu(vocab_dists_extended + attn_dists_projected)
        indices = tf.stack((batch_nums, self.X), axis=2)
        self.gold_probs = tf.gather_nd(final_dists, indices)
        self.log_losses = tf.reduce_mean(-tf.log(self.gold_probs + 1e-7))

        self.training_logits = training_decoder_output.rnn_output
        self.predicting_ids = predicting_decoder_output.sample_id
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.seq_cost = tf.contrib.seq2seq.sequence_loss(
            logits=self.training_logits, targets=self.Y, weights=masks
        )
        self.cost = self.seq_cost + self.log_losses
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        y_t = tf.argmax(self.training_logits, axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[15]:


size_layer = 128
num_layers = 2
embedded_size = 128
batch_size = 8


# In[16]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Summarization(
    size_layer, num_layers, embedded_size, len(dictionary_from), len(dictionary_to), batch_size
)
sess.run(tf.global_variables_initializer())


# In[17]:


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


# In[18]:


for EPOCH in range(10):
    lasttime = time.time()
    total_loss, total_accuracy, total_loss_test, total_accuracy_test = 0, 0, 0, 0
    train_X, train_Y = shuffle(train_X, train_Y)
    test_X, test_Y = shuffle(test_X, test_Y)
    pbar = tqdm(range(0, len(train_X), batch_size), desc="train minibatch loop")
    for k in pbar:
        batch_x, _ = pad_sentence_batch(train_X[k : min(k + batch_size, len(train_X))], PAD)
        batch_y, _ = pad_sentence_batch(train_Y[k : min(k + batch_size, len(train_X))], PAD)
        acc, loss, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X: batch_x, model.Y: batch_y},
        )
        total_loss += loss
        total_accuracy += acc
        pbar.set_postfix(cost=loss, accuracy=acc)

    pbar = tqdm(range(0, len(test_X), batch_size), desc="test minibatch loop")
    for k in pbar:
        batch_x, _ = pad_sentence_batch(test_X[k : min(k + batch_size, len(test_X))], PAD)
        batch_y, _ = pad_sentence_batch(test_Y[k : min(k + batch_size, len(test_X))], PAD)
        acc, loss = sess.run(
            [model.accuracy, model.cost], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        total_loss_test += loss
        total_accuracy_test += acc
        pbar.set_postfix(cost=loss, accuracy=acc)

    total_loss /= len(train_X) / batch_size
    total_accuracy /= len(train_X) / batch_size
    total_loss_test /= len(test_X) / batch_size
    total_accuracy_test /= len(test_X) / batch_size

    print("epoch: %d, avg loss: %f, avg accuracy: %f" % (EPOCH, total_loss, total_accuracy))
    print(
        "epoch: %d, avg loss test: %f, avg accuracy test: %f"
        % (EPOCH, total_loss_test, total_accuracy_test)
    )


# In[19]:


sess.run(model.predicting_ids, feed_dict={model.X: batch_x})


# In[20]:


batch_y


# In[ ]:
