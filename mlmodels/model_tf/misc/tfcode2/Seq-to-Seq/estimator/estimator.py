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
        batch_size,
        from_dict_size,
        to_dict_size,
        grad_clip=5.0,
    ):
        self.size_layer = size_layer
        self.num_layers = num_layers
        self.embedded_size = embedded_size
        self.grad_clip = grad_clip
        self.from_dict_size = from_dict_size
        self.to_dict_size = to_dict_size
        self.batch_size = batch_size
        self.model = tf.estimator.Estimator(self.model_fn)

    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.size_layer, reuse=reuse)

    def seq2seq(self, x_dict, reuse):
        x = x_dict["x"]
        x_seq_len = x_dict["x_len"]
        with tf.variable_scope("encoder", reuse=reuse):
            encoder_embedding = (
                tf.get_variable("encoder_embedding")
                if reuse
                else tf.get_variable(
                    "encoder_embedding",
                    [self.from_dict_size, self.embedded_size],
                    tf.float32,
                    tf.random_uniform_initializer(-1.0, 1.0),
                )
            )
            _, encoder_state = tf.nn.dynamic_rnn(
                cell=tf.nn.rnn_cell.MultiRNNCell(
                    [self.lstm_cell() for _ in range(self.num_layers)]
                ),
                inputs=tf.nn.embedding_lookup(encoder_embedding, x),
                sequence_length=x_seq_len,
                dtype=tf.float32,
            )
            encoder_state = tuple(encoder_state[-1] for _ in range(self.num_layers))
        if not reuse:
            y = x_dict["y"]
            y_seq_len = x_dict["y_len"]
            with tf.variable_scope("decoder", reuse=reuse):
                decoder_embedding = tf.get_variable(
                    "decoder_embedding",
                    [self.to_dict_size, self.embedded_size],
                    tf.float32,
                    tf.random_uniform_initializer(-1.0, 1.0),
                )
                helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=tf.nn.embedding_lookup(decoder_embedding, y),
                    sequence_length=y_seq_len,
                    time_major=False,
                )
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=tf.nn.rnn_cell.MultiRNNCell(
                        [self.lstm_cell() for _ in range(self.num_layers)]
                    ),
                    helper=helper,
                    initial_state=encoder_state,
                    output_layer=tf.layers.Dense(self.to_dict_size),
                )
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder,
                    impute_finished=True,
                    maximum_iterations=tf.reduce_max(y_seq_len),
                )
                return decoder_output.rnn_output
        else:
            with tf.variable_scope("decoder", reuse=reuse):
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=tf.get_variable("decoder_embedding"),
                    start_tokens=tf.tile(tf.constant([GO], dtype=tf.int32), [tf.shape(x)[0]]),
                    end_token=EOS,
                )
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=tf.nn.rnn_cell.MultiRNNCell(
                        [self.lstm_cell(reuse=True) for _ in range(self.num_layers)]
                    ),
                    helper=helper,
                    initial_state=encoder_state,
                    output_layer=tf.layers.Dense(self.to_dict_size, _reuse=reuse),
                )
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder,
                    impute_finished=True,
                    maximum_iterations=2 * tf.reduce_max(x_seq_len),
                )
                return decoder_output.sample_id

    def model_fn(self, features, labels, mode):
        logits = self.seq2seq(features, reuse=False)
        predictions = self.seq2seq(features, reuse=True)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        y_seq_len = features["y_len"]
        masks = tf.sequence_mask(y_seq_len, tf.reduce_max(y_seq_len), dtype=tf.float32)
        loss_op = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=labels, weights=masks)
        params = tf.trainable_variables()
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        train_op = tf.train.AdamOptimizer().apply_gradients(
            zip(clipped_gradients, params), global_step=tf.train.get_global_step()
        )
        acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions)
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={"accuracy": acc_op},
        )
        return estim_specs


# In[8]:


size_layer = 256
num_layers = 2
embedded_size = 256
batch_size = len(text_from)
model = Chatbot(
    size_layer,
    num_layers,
    embedded_size,
    batch_size,
    vocabulary_size_from + 4,
    vocabulary_size_to + 4,
)


# In[9]:


def str_idx(corpus, dic):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            try:
                ints.append(dic[k])
            except Exception as e:
                print(e)
                ints.append(UNK)
        X.append(ints)
    return X


X = str_idx(text_from, dictionary_from)
Y = str_idx(text_to, dictionary_to)


# In[10]:


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return np.array(padded_seqs).astype(np.int32), np.array(seq_lens).astype(np.int32)


# In[13]:


batch_x, seq_x = pad_sentence_batch(X, PAD)
batch_y, seq_y = pad_sentence_batch(Y, PAD)


# In[14]:


input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": batch_x, "x_len": seq_x, "y": batch_y, "y_len": seq_y},
    y=batch_y,
    batch_size=batch_size,
    num_epochs=100,
    shuffle=True,
)
model.model.train(input_fn)


# In[ ]:
