#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense

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


# In[3]:


with open("data/from", "r") as fopen:
    text_from = fopen.read().lower().split("\n")
with open("data/to", "r") as fopen:
    text_to = fopen.read().lower().split("\n")
print("len from: %d, len to: %d" % (len(text_from), len(text_to)))


# In[4]:


concat_from = " ".join(text_from).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(
    concat_from, vocabulary_size_from
)
print("vocab from size: %d" % (vocabulary_size_from))
print("Most common words", count_from[3:10])
print("Sample data", data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])


# In[5]:


concat_to = " ".join(text_to).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, vocabulary_size_to)
print("vocab to size: %d" % (vocabulary_size_to))
print("Most common words", count_to[3:10])
print("Sample data", data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])


# In[6]:


GO = dictionary_from["GO"]
PAD = dictionary_from["PAD"]
EOS = dictionary_from["EOS"]
UNK = dictionary_from["UNK"]


# In[7]:


class Chatbot:
    def __init__(
        self,
        size_layer,
        num_layers,
        embedded_size,
        from_dict_size,
        to_dict_size,
        learning_rate,
        batch_size,
        dropout=0.5,
    ):
        def lstm_cell(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(size_layer, reuse=reuse)

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
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])
        # encoder
        encoder_embeddings = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        encoder_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
        encoder_dropout = tf.contrib.rnn.DropoutWrapper(encoder_cells, output_keep_prob=0.5)
        self.encoder_out, self.encoder_state = tf.nn.dynamic_rnn(
            cell=encoder_dropout,
            inputs=encoder_embedded,
            sequence_length=self.X_seq_len,
            dtype=tf.float32,
        )

        self.encoder_state = tuple(self.encoder_state[-1] for _ in range(num_layers))
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        # decoder
        decoder_embeddings = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))
        decoder_cell = attention(self.encoder_out, self.X_seq_len)
        dense_layer = Dense(to_dict_size)
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=tf.nn.embedding_lookup(decoder_embeddings, decoder_input),
            sequence_length=self.Y_seq_len,
            time_major=False,
        )
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=training_helper,
            initial_state=decoder_cell.zero_state(batch_size, tf.float32).clone(
                cell_state=self.encoder_state
            ),
            output_layer=dense_layer,
        )
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
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
            maximum_iterations=2 * tf.reduce_max(self.X_seq_len),
        )
        self.training_logits = training_decoder_output.rnn_output
        self.predicting_ids = predicting_decoder_output.sample_id
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(
            logits=self.training_logits, targets=self.Y, weights=masks
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


# In[8]:


size_layer = 256
num_layers = 2
embedded_size = 256
learning_rate = 0.001
batch_size = 32
epoch = 50


# In[9]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Chatbot(
    size_layer,
    num_layers,
    embedded_size,
    vocabulary_size_from + 4,
    vocabulary_size_to + 4,
    learning_rate,
    batch_size,
)
sess.run(tf.global_variables_initializer())


# In[10]:


def str_idx(corpus, dic):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            try:
                ints.append(dic[k])
            except Exception as e:
                print(e)
                ints.append(2)
        X.append(ints)
    return X


# In[11]:


X = str_idx(text_from, dictionary_from)
Y = str_idx(text_to, dictionary_to)


# In[12]:


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


# In[13]:


for i in range(epoch):
    total_loss = 0
    for k in range(0, (len(text_from) // batch_size) * batch_size, batch_size):
        batch_x, seq_x = pad_sentence_batch(X[k : k + batch_size], PAD)
        batch_y, seq_y = pad_sentence_batch(Y[k : k + batch_size], PAD)
        loss, _ = sess.run(
            [model.cost, model.optimizer],
            feed_dict={
                model.X: batch_x,
                model.Y: batch_y,
                model.X_seq_len: seq_x,
                model.Y_seq_len: seq_y,
            },
        )
        total_loss += loss
    total_loss /= len(text_from) // batch_size
    print("epoch: %d, avg loss: %f" % (i + 1, total_loss))


# In[14]:


def predict(X, Y, from_dict, to_dict, batch_size):
    out_indices = sess.run(
        model.predicting_ids, {model.X: [X] * batch_size, model.X_seq_len: [len(X)] * batch_size}
    )[0]

    print("FROM")
    print("IN:", [i for i in X])
    print("WORD:", " ".join([from_dict[i] for i in X]))
    print("\nTO")
    print("OUT:", [i for i in out_indices])
    print("WORD:", " ".join([to_dict[i] for i in out_indices]))
    print("ACTUAL REPLY:", " ".join([to_dict[i] for i in Y]))


# In[15]:


predict(X[2], Y[2], rev_dictionary_from, rev_dictionary_to, batch_size)


# In[ ]:
