#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# In[2]:


def parse_text(file_path):
    with open(file_path) as f:
        text = f.read()

    char2idx = {c: i + 3 for i, c in enumerate(set(text))}
    char2idx["<pad>"] = 0
    char2idx["<start>"] = 1
    char2idx["<end>"] = 2

    vector = np.array([char2idx[char] for char in list(text)])
    return vector, char2idx


# In[3]:


vector, char2idx = parse_text("shakespeare.txt")
idx2char = {i: c for c, i in char2idx.items()}


# In[4]:


batch_size = 128
seq_len = 100
hidden_dim = 128
n_layers = 2
beam_width = 5
clip_norm = 100.0
skip = 20


# In[5]:


def cell_fn():
    return tf.nn.rnn_cell.ResidualWrapper(
        tf.nn.rnn_cell.GRUCell(hidden_dim, kernel_initializer=tf.orthogonal_initializer())
    )


def multi_cell_fn():
    return tf.nn.rnn_cell.MultiRNNCell([cell_fn() for _ in range(n_layers)])


def clip_grads(loss):
    variables = tf.trainable_variables()
    grads = tf.gradients(loss, variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm)
    return zip(clipped_grads, variables)


class Model:
    def __init__(self, seq_len, vocab_size, hidden_dim):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.batch_size = tf.shape(self.X)[0]
        encoder_embeddings = tf.Variable(tf.random_uniform([vocab_size, hidden_dim], -1, 1))
        cells = multi_cell_fn()
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=tf.nn.embedding_lookup(encoder_embeddings, self.X),
            sequence_length=tf.count_nonzero(self.X, 1, dtype=tf.int32),
        )
        dense_layer = tf.layers.Dense(vocab_size)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cells,
            helper=helper,
            initial_state=cells.zero_state(self.batch_size, tf.float32),
            output_layer=dense_layer,
        )

        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
        self.logits = decoder_output.rnn_output

        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=cells,
            embedding=encoder_embeddings,
            start_tokens=tf.tile(tf.constant([char2idx["<start>"]], dtype=tf.int32), [1]),
            end_token=char2idx["<end>"],
            initial_state=tf.contrib.seq2seq.tile_batch(
                cells.zero_state(1, tf.float32), beam_width
            ),
            beam_width=beam_width,
            output_layer=dense_layer,
        )

        decoder_out, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, maximum_iterations=seq_len
        )

        self.predict = decoder_out.predicted_ids[:, :, 0]

        self.cost = tf.reduce_mean(
            tf.contrib.seq2seq.sequence_loss(
                logits=self.logits, targets=self.Y, weights=tf.to_float(tf.ones_like(self.Y))
            )
        )
        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer().apply_gradients(
            clip_grads(self.cost), global_step=self.global_step
        )


# In[6]:


def start_sentence(x):
    _x = np.full([x.shape[0], 1], char2idx["<start>"])
    return np.concatenate([_x, x], 1)


def end_sentence(x):
    _x = np.full([x.shape[0], 1], char2idx["<end>"])
    return np.concatenate([x, _x], 1)


batches = []
for i in range(0, len(vector) - seq_len, skip):
    batches.append(vector[i : i + seq_len])
X = np.array(batches)


# In[7]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(seq_len, len(char2idx), hidden_dim)
sess.run(tf.global_variables_initializer())


# In[8]:


for e in range(10):
    lasttime = time.time()
    train_loss, test_loss = 0, 0
    pbar = tqdm(range(0, len(X), batch_size), desc="train minibatch loop")
    for i in pbar:
        batch_x = X[i : min(i + batch_size, len(X))]
        batch_y = end_sentence(batch_x)
        batch_x = start_sentence(batch_x)
        loss, _ = sess.run(
            [model.cost, model.optimizer], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        assert not np.isnan(loss)
        train_loss += loss
        pbar.set_postfix(cost=loss)

    batch_x = start_sentence(X[:batch_size])
    ints = sess.run(model.predict, feed_dict={model.X: batch_x})[0]
    print("\n" + "".join([idx2char[i] for i in ints]) + "\n")


# In[ ]:
