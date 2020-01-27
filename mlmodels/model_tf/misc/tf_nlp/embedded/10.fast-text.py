#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import itertools
import os
import re

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from tqdm import tqdm

from unidecode import unidecode
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = ""


# In[2]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[3]:


def _pad_sequence(
    sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None
):
    sequence = iter(sequence)
    if pad_left:
        sequence = itertools.chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = itertools.chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(
    sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None
):
    """
    generate ngrams

    Parameters
    ----------
    sequence : list of str
        list of tokenize words
    n : int
        ngram size

    Returns
    -------
    ngram: list
    """
    sequence = _pad_sequence(sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol)
    history = []
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def generator(word, ngram=(2, 3)):
    return ["".join(i) for n in ngram for i in ngrams(word, n)]


def build_dict(word_counter, vocab_size=50000):
    count = [["PAD", 0], ["UNK", 1], ["START", 2], ["END", 3]]
    count.extend(word_counter.most_common(vocab_size))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    return dictionary, {word: idx for idx, word in dictionary.items()}


def doc2num(word_list, dictionary):
    word_array = []
    for word in word_list:
        words = generator(word)
        word_array.append([dictionary.get(word, 1) for word in words])
    return word_array


def build_word_array(sentences, vocab_size):
    word_counter, word_list, num_lines, num_words = counter_words(sentences)
    dictionary, rev_dictionary = build_dict(word_counter, vocab_size)
    word_array = doc2num(word_list, dictionary)
    return word_array, dictionary, rev_dictionary, num_lines, num_words


def build_training_set(word_array, maxlen=100):
    num_words = len(word_array)
    maxlen = max([len(i) for i in word_array]) if not maxlen else maxlen
    x = np.zeros((num_words - 4, maxlen, 4), dtype=np.int32)
    y = np.zeros((num_words - 4, maxlen), dtype=np.int32)
    shift = [-2, -1, 1, 2]
    for idx in range(2, num_words - 2):
        y[idx - 2, : len(word_array[idx])] = word_array[idx][:maxlen]
        for no, s in enumerate(shift):
            x[idx - 2, : len(word_array[idx + s]), no] = word_array[idx + s][:maxlen]
    return x, y


def counter_words(sentences):
    word_counter = collections.Counter()
    word_list = []
    num_lines, num_words = (0, 0)
    for i in sentences:
        words = re.sub("[^'\"A-Za-z\-<> ]+", " ", unidecode(i))
        word_list.append(words)
        words = generator(words)
        word_counter.update(words)
        num_lines += 1
        num_words += len(words)
    return word_counter, word_list, num_lines, num_words


# In[4]:


sentences = ["<%s>" % (w) for w in " ".join(trainset.data).split()]


# In[5]:


get_ipython().run_cell_magic(
    "time",
    "",
    "word_array, dictionary, rev_dictionary, num_lines, num_words = build_word_array(sentences,\n                                                                                vocab_size=1000000)",
)


# In[6]:


len(dictionary)


# In[7]:


X, Y = build_training_set(word_array[:32])


# In[8]:


graph_params = {
    "batch_size": 128,
    "vocab_size": len(dictionary),
    "embed_size": 1024,
    "hid_size": 1024,
    "neg_samples": 128,
    "learn_rate": 0.01,
    "momentum": 0.9,
    "embed_noise": 0.1,
    "hid_noise": 0.3,
    "epoch": 5,
    "optimizer": "Momentum",
}
maxlen = 100


# In[13]:


class Model:
    def __init__(self, graph_params):
        g_params = graph_params
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.X = tf.placeholder(tf.int64, shape=[None, None, 4])
        self.Y = tf.placeholder(tf.int64, shape=[None, None])
        length_X = tf.count_nonzero(self.X, 1)
        length_Y = tf.count_nonzero(self.Y, 1)

        w_m2, w_m1, w_p1, w_p2 = tf.unstack(self.X, axis=2)
        self.embed_weights = tf.Variable(
            tf.random_uniform(
                [g_params["vocab_size"], g_params["embed_size"]],
                -g_params["embed_noise"],
                g_params["embed_noise"],
            )
        )
        y = tf.argmax(tf.nn.embedding_lookup(self.embed_weights, self.Y), axis=-1)
        embed_m2 = tf.reduce_mean(tf.nn.embedding_lookup(self.embed_weights, w_m2), axis=1)
        embed_m1 = tf.reduce_mean(tf.nn.embedding_lookup(self.embed_weights, w_m1), axis=1)
        embed_p1 = tf.reduce_mean(tf.nn.embedding_lookup(self.embed_weights, w_p1), axis=1)
        embed_p2 = tf.reduce_mean(tf.nn.embedding_lookup(self.embed_weights, w_p2), axis=1)
        embed_stack = tf.concat([embed_m2, embed_m1, embed_p1, embed_p2], 1)
        hid_weights = tf.Variable(
            tf.random_normal(
                [g_params["embed_size"] * 4, g_params["hid_size"]],
                stddev=g_params["hid_noise"] / (g_params["embed_size"] * 4) ** 0.5,
            )
        )
        hid_bias = tf.Variable(tf.zeros([g_params["hid_size"]]))
        hid_out = tf.nn.tanh(tf.matmul(embed_stack, hid_weights) + hid_bias)
        self.nce_weights = tf.Variable(
            tf.random_normal(
                [g_params["vocab_size"], g_params["hid_size"]],
                stddev=1.0 / g_params["hid_size"] ** 0.5,
            )
        )
        nce_bias = tf.Variable(tf.zeros([g_params["vocab_size"]]))
        self.cost = tf.reduce_mean(
            tf.nn.nce_loss(
                self.nce_weights,
                nce_bias,
                inputs=hid_out,
                labels=y,
                num_sampled=g_params["neg_samples"],
                num_classes=g_params["vocab_size"],
                num_true=maxlen,
                remove_accidental_hits=True,
            )
        )
        if g_params["optimizer"] == "RMSProp":
            self.optimizer = tf.train.RMSPropOptimizer(g_params["learn_rate"]).minimize(self.cost)
        elif g_params["optimizer"] == "Momentum":
            self.optimizer = tf.train.MomentumOptimizer(
                g_params["learn_rate"], g_params["momentum"]
            ).minimize(self.cost)
        elif g_params["optimizer"] == "Adam":
            self.optimizer = tf.train.AdamOptimizer(g_params["learn_rate"]).minimize(self.cost)
        else:
            print("Optimizer not supported,exit.")
        self.sess.run(tf.global_variables_initializer())

    def train(self, train, epoch, batch_size):
        for i in range(epoch):
            pbar = tqdm(range(0, len(train), batch_size), desc="train minibatch loop")
            for batch in pbar:
                X, Y = build_training_set(
                    train[batch : min(batch + batch_size, len(train))], maxlen=maxlen
                )
                X, Y = shuffle(X, Y)
                feed_dict = {self.X: X, self.Y: Y}
                _, loss = self.sess.run([self.optimizer, self.cost], feed_dict=feed_dict)
                pbar.set_postfix(cost=loss)

        return self.embed_weights.eval(), self.nce_weights.eval()


# In[14]:


model = Model(graph_params)


# In[15]:


embed_weights, nce_weights = model.train(
    word_array, graph_params["epoch"], graph_params["batch_size"]
)


# In[33]:


def doc2num(word_list, dictionary, ngrams=(2, 3)):
    word_array = []
    for word in word_list:
        words = generator(word, ngram=ngrams)
        word_array.append([dictionary.get(word, 1) for word in words])
    return word_array


# In[61]:


word = "eat"
word_array = doc2num(["<%s>" % (word)], dictionary)[0]
eat_vector = np.array([nce_weights[i] for i in word_array]).sum(axis=0)


# In[62]:


words = ["ate", "eating", "shitting", "giving", "water"]
pools = []
for word in words:
    word = filter(None, word.split())
    pools.append("".join(["<%s>" % (w) for w in word]))
word_array = doc2num(pools, dictionary)
outside_array = []
for arr in word_array:
    outside_array.append(np.array([nce_weights[i] for i in arr]).sum(axis=0))


# In[63]:


outside_array


# In[64]:


# In[65]:


nn = NearestNeighbors(3, metric="cosine").fit(outside_array)
distances, idx = nn.kneighbors(eat_vector.reshape((1, -1)))
word_list = []
for i in range(1, idx.shape[1]):
    word_list.append([words[idx[0, i]], 1 - distances[0, i]])
word_list


# In[ ]:
