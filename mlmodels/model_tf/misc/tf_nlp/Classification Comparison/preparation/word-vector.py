#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import os
import random
import re
import time

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# In[2]:


def clearstring(string):
    string = re.sub("[^'\"A-Za-z0-9 ]+", "", string)
    string = string.split(" ")
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = [y for y in string if len(y) > 3 and y.find("nbsp") < 0]
    return " ".join(string)


def read_data():
    list_folder = os.listdir("data/")
    label = list_folder
    label.sort()
    outer_string, outer_label = [], []
    for i in range(len(list_folder)):
        list_file = os.listdir("data/" + list_folder[i])
        strings = []
        for x in range(len(list_file)):
            with open("data/" + list_folder[i] + "/" + list_file[x], "r") as fopen:
                strings += fopen.read().split("\n")
        strings = list(filter(None, strings))
        for k in range(len(strings)):
            strings[k] = clearstring(strings[k])
        labels = [i] * len(strings)
        outer_string += strings
        outer_label += labels

    dataset = np.array([outer_string, outer_label])
    dataset = dataset.T
    np.random.shuffle(dataset)

    string = []
    for i in range(dataset.shape[0]):
        string += dataset[i][0].split()

    return string, dataset, label


# In[3]:


def build_dataset(words, n_words):
    count = [["UNK", -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    global data
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index : data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            for word in data[:span]:
                buffer.append(word)
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# In[4]:


vocabulary, dataset, label = read_data()
print("example 10 words:", vocabulary[:10])
print("label dataset:", label)
print("size corpus:", len(vocabulary))
vocabulary_size = len(list(set(vocabulary)))
print("size of unique words:", vocabulary_size)
dimension = 300
skip_window = 1
num_skips = 2
iteration_train_vectors = 100
batch_size = 64
location = os.getcwd()


# In[5]:


data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
print("Most common words (+UNK)", count[:5])
print("Sample data", data[:10], [reverse_dictionary[i] for i in data[:10]])


# In[6]:


valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


# In[7]:


graph = tf.Graph()

with graph.as_default():

    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device("/cpu:0"):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, dimension], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, dimension], stddev=1.0 / np.sqrt(dimension))
        )
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=batch_size / 2,
                num_classes=vocabulary_size,
            )
        )
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        init = tf.global_variables_initializer()

num_steps = 100000


# In[8]:


with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1 : top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    embedding_vals = session.run(embeddings)


# In[ ]:


with open("vector-emotion.p", "wb") as fopen:
    pickle.dump(embedding_vals, fopen)
