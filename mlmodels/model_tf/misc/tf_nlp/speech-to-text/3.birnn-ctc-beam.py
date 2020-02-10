#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import librosa

# In[2]:


wav_files = [f for f in os.listdir("./data") if f.endswith(".wav")]
text_files = [f for f in os.listdir("./data") if f.endswith(".txt")]


# In[3]:


inputs, targets = [], []
for (wav_file, text_file) in tqdm(zip(wav_files, text_files), total=len(wav_files), ncols=80):
    path = "./data/" + wav_file
    try:
        y, sr = librosa.load(path, sr=None)
    except:
        continue
    inputs.append(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=int(1e-1 * sr)).T)
    with open("./data/" + text_file) as f:
        targets.append(f.read())


# In[4]:


inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, dtype="float32", padding="post")

chars = list(set([c for target in targets for c in target]))
num_classes = len(chars) + 1

idx2char = {idx: char for idx, char in enumerate(chars)}
char2idx = {char: idx for idx, char in idx2char.items()}

targets = [[char2idx[c] for c in target] for target in targets]


# In[5]:


def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


# In[6]:


class Model:
    def __init__(self, num_layers, size_layers, learning_rate, num_features, dropout=1.0):
        self.X = tf.placeholder(tf.float32, [None, None, num_features])
        self.Y = tf.sparse_placeholder(tf.int32)
        seq_lens = tf.count_nonzero(tf.reduce_sum(self.X, -1), 1, dtype=tf.int32)

        def cells(size, reuse=False):
            return tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(), reuse=reuse),
                state_keep_prob=dropout,
                output_keep_prob=dropout,
            )

        features = self.X
        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cells(size_layers),
                cell_bw=cells(size_layers),
                inputs=features,
                sequence_length=seq_lens,
                dtype=tf.float32,
                scope="bidirectional_rnn_%d" % (n),
            )
            features = tf.concat((out_fw, out_bw), 2)

        logits = tf.layers.dense(features, num_classes)
        time_major = tf.transpose(logits, [1, 0, 2])
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(time_major, seq_lens)
        decoded = tf.to_int32(decoded[0])
        self.preds = tf.sparse.to_dense(decoded)
        self.cost = tf.reduce_mean(
            tf.nn.ctc_loss(self.Y, time_major, seq_lens, ignore_longer_outputs_than_inputs=True)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


# In[7]:


tf.reset_default_graph()
sess = tf.InteractiveSession()

size_layers = 128
learning_rate = 1e-3
num_layers = 2

model = Model(num_layers, size_layers, learning_rate, inputs.shape[2])
sess.run(tf.global_variables_initializer())


# In[8]:


batch_size = 32

for e in range(50):
    lasttime = time.time()
    pbar = tqdm(range(0, len(inputs), batch_size), desc="minibatch loop", ncols=80)
    for i in pbar:
        batch_x = inputs[i : min(i + batch_size, len(inputs))]
        batch_y = sparse_tuple_from(targets[i : min(i + batch_size, len(inputs))])
        _, cost = sess.run(
            [model.optimizer, model.cost], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        pbar.set_postfix(cost=cost)


# In[9]:


random_index = random.randint(0, len(targets) - 1)
batch_x = inputs[random_index : random_index + 1]
print("real:", "".join([idx2char[no] for no in targets[random_index : random_index + 1][0]]))
batch_y = sparse_tuple_from(targets[random_index : random_index + 1])
pred = sess.run(model.preds, feed_dict={model.X: batch_x})[0]
print("predicted:", "".join([idx2char[no] for no in pred]))


# In[ ]:
