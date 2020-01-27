#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import tensorflow as tf
from scipy.io.wavfile import write
from tqdm import tqdm

from utils import *

# In[2]:


def prenet(inputs, num_units=None, is_training=True, scope="prenet"):
    if num_units is None:
        num_units = [embed_size, embed_size // 2]
    with tf.variable_scope(scope):
        outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu, name="dense1")
        outputs = tf.layers.dropout(
            outputs, rate=dropout_rate, training=is_training, name="dropout1"
        )
        outputs = tf.layers.dense(outputs, units=num_units[1], activation=tf.nn.relu, name="dense2")
        outputs = tf.layers.dropout(
            outputs, rate=dropout_rate, training=is_training, name="dropout2"
        )
    return outputs


def highwaynet(inputs, num_units=None, scope="highwaynet"):
    if not num_units:
        num_units = inputs.get_shape()[-1]
    with tf.variable_scope(scope):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
        T = tf.layers.dense(
            inputs,
            units=num_units,
            activation=tf.nn.sigmoid,
            bias_initializer=tf.constant_initializer(-1.0),
            name="dense2",
        )
        outputs = H * T + inputs * (1.0 - T)
    return outputs


def conv1d_banks(inputs, K=16, is_training=True, scope="conv1d_banks"):
    with tf.variable_scope(scope):
        outputs = tf.layers.conv1d(inputs, embed_size // 2, 1, padding="SAME")
        for k in range(2, K + 1):
            with tf.variable_scope("num_{}".format(k)):
                output = tf.layers.conv1d(inputs, embed_size // 2, k, padding="SAME")
                outputs = tf.concat((outputs, output), -1)
        outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=is_training))
    return outputs


class Model:
    def __init__(self, num_layers, size_layers, learning_rate=1e-3, dropout=1.0):
        self.X = tf.placeholder(tf.int32, (None, None))
        self.training = tf.placeholder(tf.bool, None)
        lookup_table = tf.get_variable(
            "lookup_table",
            dtype=tf.float32,
            shape=[len(vocab), size_layers],
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        )
        lookup_table = tf.concat((tf.zeros(shape=[1, size_layers]), lookup_table[1:, :]), 0)
        forward = tf.nn.embedding_lookup(lookup_table, self.X)
        self.Y = tf.placeholder(tf.float32, (None, None, n_mels * resampled))
        self.decoder_inputs = tf.concat((tf.zeros_like(self.Y[:, :1, :]), self.Y[:, :-1, :]), 1)
        self.decoder_inputs = self.decoder_inputs[:, :, -n_mels:]
        self.Z = tf.placeholder(tf.float32, (None, None, fourier_window_size // 2 + 1))

        batch_size = tf.shape(self.X)[0]
        seq_lens = tf.count_nonzero(tf.reduce_sum(self.decoder_inputs, -1), 1, dtype=tf.int32) + 1

        def cells(reuse=False):
            return tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(
                    size_layers, initializer=tf.orthogonal_initializer(), reuse=reuse
                ),
                state_keep_prob=dropout,
                output_keep_prob=dropout,
            )

        def attention(encoder_out, seq_len, reuse=False):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=size_layers, memory=encoder_out, memory_sequence_length=seq_len
            )
            return tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([cells(reuse) for _ in range(num_layers)]),
                attention_mechanism=attention_mechanism,
                attention_layer_size=size_layers,
                alignment_history=True,
            )

        encoder_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        encoder_out, encoder_state = tf.nn.dynamic_rnn(
            cell=encoder_cells, inputs=forward, sequence_length=seq_lens, dtype=tf.float32
        )

        encoder_state = tuple(encoder_state[-1] for _ in range(num_layers))
        decoder_cell = attention(encoder_out, seq_lens)
        dense_layer = tf.layers.Dense(n_mels * resampled)

        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=self.decoder_inputs, sequence_length=seq_lens, time_major=False
        )
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=training_helper,
            initial_state=decoder_cell.zero_state(batch_size, tf.float32).clone(
                cell_state=encoder_state
            ),
            output_layer=dense_layer,
        )
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=training_decoder,
            impute_finished=True,
            maximum_iterations=tf.reduce_max(seq_lens),
        )

        self.Y_hat = training_decoder_output.rnn_output
        out_decoder2 = tf.reshape(self.Y_hat, [tf.shape(self.Y_hat)[0], -1, n_mels])
        dec = conv1d_banks(out_decoder2, K=decoder_num_banks, is_training=self.training)
        dec = tf.layers.max_pooling1d(dec, pool_size=2, strides=1, padding="same")
        dec = tf.layers.conv1d(dec, embed_size // 2, 3, name="decoder-conv1-1", padding="SAME")
        dec = tf.nn.relu(tf.layers.batch_normalization(dec, training=self.training))
        dec = tf.layers.conv1d(dec, embed_size // 2, 3, name="decoder-conv1-2", padding="SAME")
        dec = tf.layers.batch_normalization(dec, training=self.training)
        dec = tf.layers.dense(dec, embed_size // 2)
        for i in range(4):
            dec = highwaynet(
                dec, num_units=embed_size // 2, scope="decoder-highwaynet-{}".format(i)
            )
        with tf.variable_scope("decoder-gru", reuse=False):
            cell = tf.contrib.rnn.GRUCell(embed_size // 2)
            cell_bw = tf.contrib.rnn.GRUCell(embed_size // 2)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, dec, dtype=tf.float32)
            outputs = tf.concat(outputs, 2)
        self.Z_hat = tf.layers.dense(outputs, 1 + fourier_window_size // 2)
        self.loss1 = tf.reduce_mean(tf.abs(self.Y_hat - self.Y))
        self.loss2 = tf.reduce_mean(tf.abs(self.Z_hat - self.Z))
        self.loss = self.loss1 + self.loss2
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


# In[3]:


tf.reset_default_graph()
sess = tf.InteractiveSession()

size_layers = 128
learning_rate = 1e-3
num_layers = 2

model = Model(num_layers, size_layers, learning_rate)
sess.run(tf.global_variables_initializer())


# In[4]:


paths, lengths, texts, raw_texts = [], [], [], []
text_files = [f for f in os.listdir("mel") if f.endswith(".npy")]
for fpath in text_files:
    with open("%s/%s" % (path, fpath.replace("npy", "txt"))) as fopen:
        text = fopen.read()
    paths.append(fpath.replace(".npy", ""))
    text = text_normalize(text)
    raw_texts.append(text)
    text = text + "E"
    texts.append(np.array([char2idx[char] for char in text], np.int32))
    lengths.append(len(text))


# In[5]:


def dynamic_batching(paths):
    files, max_y, max_z = [], 0, 0
    for n in range(len(paths)):
        files.append(get_cached(paths[n]))
        if files[-1][0].shape[0] > max_y:
            max_y = files[-1][0].shape[0]
        if files[-1][1].shape[0] > max_z:
            max_z = files[-1][1].shape[0]
    return files, max_y, max_z


# In[6]:


EPOCH = 30
for i in range(EPOCH):
    pbar = tqdm(range(0, len(paths), batch_size), desc="minibatch loop")
    for k in pbar:
        index = min(k + batch_size, len(paths))
        files, max_y, max_z = dynamic_batching(paths[k:index])
        max_x = max(lengths[k:index])
        batch_x = np.zeros((batch_size, max_x))
        batch_y = np.zeros((batch_size, max_y, n_mels * resampled))
        batch_z = np.zeros((batch_size, max_z, fourier_window_size // 2 + 1))
        for n in range(len(files)):
            batch_x[n, :] = np.pad(
                texts[k + n], ((0, max_x - texts[k + n].shape[0])), mode="constant"
            )
            batch_y[n, :, :] = np.pad(
                files[n][0], ((0, max_y - files[n][0].shape[0]), (0, 0)), mode="constant"
            )
            batch_z[n, :, :] = np.pad(
                files[n][1], ((0, max_z - files[n][1].shape[0]), (0, 0)), mode="constant"
            )
        _, cost = sess.run(
            [model.optimizer, model.loss],
            feed_dict={model.X: batch_x, model.Y: batch_y, model.Z: batch_z, model.training: True},
        )
        pbar.set_postfix(cost=cost)


# In[7]:


y_hat = np.zeros((1, 50, n_mels * resampled), np.float32)
for j in tqdm(range(50)):
    _y_hat = sess.run(model.Y_hat, {model.X: [texts[0]], model.Y: y_hat})
    y_hat[:, j, :] = _y_hat[:, j, :]


# In[8]:


mags = sess.run(model.Z_hat, {model.Y_hat: y_hat, model.training: False})


# In[9]:


audio = spectrogram2wav(mags[0])


# In[10]:


print("saving: %s" % (raw_texts[0]))
write(os.path.join("test.wav"), sample_rate, audio)


# In[ ]:
