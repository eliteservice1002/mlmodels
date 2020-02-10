#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

sns.set()


# In[2]:


def get_vocab(file, lower=False):
    with open(file, "r") as fopen:
        data = fopen.read()
    if lower:
        data = data.lower()
    vocab = list(set(data))
    return data, vocab


def embed_to_onehot(data, vocab):
    onehot = np.zeros((len(data), len(vocab)), dtype=np.float32)
    for i in range(len(data)):
        onehot[i, vocab.index(data[i])] = 1.0
    return onehot


# In[3]:


text, text_vocab = get_vocab("shakespeare.txt", lower=False)
onehot = embed_to_onehot(text, text_vocab)


# In[4]:


len(text_vocab)


# In[5]:


learning_rate = 0.01
batch_size = 32
sequence_length = 64
epoch = 3000
num_layers = 2
size_layer = 256
possible_batch_id = range(len(text) - sequence_length - 1)


# In[6]:


class Model:
    def __init__(self, num_layers, size_layer, dimension, sequence_length, learning_rate):
        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(size_layer, sequence_length)

        self.X = tf.placeholder(tf.float32, (None, None, dimension))
        self.Y = tf.placeholder(tf.float32, (None, None, dimension))
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=size_layer, memory=self.X
        )
        self.rnn_cells = tf.contrib.seq2seq.AttentionWrapper(
            cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)]),
            attention_mechanism=attention_mechanism,
            attention_layer_size=size_layer,
        )
        self.initial_state = self.rnn_cells.zero_state(
            dtype=tf.float32, batch_size=tf.shape(self.X)[0]
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            self.rnn_cells, self.X, initial_state=self.initial_state, dtype=tf.float32
        )
        rnn_W = tf.Variable(tf.random_normal((size_layer, dimension)))
        rnn_B = tf.Variable(tf.random_normal([dimension]))
        self.logits = tf.matmul(tf.reshape(self.outputs, [-1, size_layer]), rnn_W) + rnn_B
        y_batch_long = tf.reshape(self.Y, [-1, dimension])
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_batch_long)
        )
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(y_batch_long, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        seq_shape = tf.shape(self.outputs)
        self.final_outputs = tf.reshape(
            tf.nn.softmax(self.logits), (seq_shape[0], seq_shape[1], dimension)
        )


# In[7]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(num_layers, size_layer, len(text_vocab), sequence_length, learning_rate)
sess.run(tf.global_variables_initializer())


# In[8]:


split_text = text.split()
tag = split_text[np.random.randint(0, len(split_text))]
print(tag)


# In[9]:


def train_random_sequence():
    LOST, ACCURACY = [], []
    pbar = tqdm(range(epoch), desc="epoch")
    batch_x = np.zeros((batch_size, sequence_length, len(text_vocab)))
    batch_y = np.zeros((batch_size, sequence_length, len(text_vocab)))
    batch_id = random.sample(possible_batch_id, batch_size)
    for n in range(sequence_length):
        id1 = [k + n for k in batch_id]
        id2 = [k + n + 1 for k in batch_id]
        batch_x[:, n, :] = onehot[id1, :]
        batch_y[:, n, :] = onehot[id2, :]
    last_state, _ = sess.run(
        [model.last_state, model.optimizer], feed_dict={model.X: batch_x, model.Y: batch_y}
    )
    for i in pbar:
        last_time = time.time()
        batch_x = np.zeros((batch_size, sequence_length, len(text_vocab)))
        batch_y = np.zeros((batch_size, sequence_length, len(text_vocab)))
        batch_id = random.sample(possible_batch_id, batch_size)
        for n in range(sequence_length):
            id1 = [k + n for k in batch_id]
            id2 = [k + n + 1 for k in batch_id]
            batch_x[:, n, :] = onehot[id1, :]
            batch_y[:, n, :] = onehot[id2, :]
        last_state, _, loss, accuracy = sess.run(
            [model.last_state, model.optimizer, model.cost, model.accuracy],
            feed_dict={model.X: batch_x, model.Y: batch_y, model.initial_state: last_state},
        )
        ACCURACY.append(accuracy)
        LOST.append(loss)
        pbar.set_postfix(cost=loss, accuracy=accuracy)
    return LOST, ACCURACY


# In[10]:


LOST, ACCURACY = train_random_sequence()


# In[11]:


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
EPOCH = np.arange(len(LOST))
plt.plot(EPOCH, LOST)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.subplot(1, 2, 2)
plt.plot(EPOCH, ACCURACY)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()


# In[14]:


sentence_generated = tag
onehot = embed_to_onehot(tag, text_vocab)


# In[26]:


def generate_based_sequence(length_sentence, argmax=False):
    sentence_generated = tag
    onehot = embed_to_onehot(tag, text_vocab)
    batch_x = np.zeros((1, 1, len(text_vocab)))
    batch_x[:, 0, :] = onehot[0, :]
    last_state, prob = sess.run(
        [model.last_state, model.final_outputs], feed_dict={model.X: batch_x}
    )
    for i in range(1, len(tag), 1):
        batch_x = np.zeros((1, 1, len(text_vocab)))
        batch_x[:, 0, :] = onehot[i, :]
        last_state, prob = sess.run(
            [model.last_state, model.final_outputs],
            feed_dict={model.X: batch_x, model.initial_state: last_state},
        )

    for i in range(length_sentence):
        if argmax:
            char = np.argmax(prob[0][0])
        else:
            char = np.random.choice(range(len(text_vocab)), p=prob[0][0])
        element = text_vocab[char]
        sentence_generated += element
        onehot = embed_to_onehot(element, text_vocab)
        batch_x = np.zeros((1, 1, len(text_vocab)))
        batch_x[:, 0, :] = onehot[0, :]
        last_state, prob = sess.run(
            [model.last_state, model.final_outputs],
            feed_dict={model.X: batch_x, model.initial_state: last_state},
        )

    return sentence_generated


# In[27]:


print(generate_based_sequence(1000, True))


# In[17]:


print(generate_based_sequence(1000, False))


# In[ ]:
