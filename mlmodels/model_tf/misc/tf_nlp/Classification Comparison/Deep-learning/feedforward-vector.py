#!/usr/bin/env python
# coding: utf-8

# In[15]:


import collections
import os
import random
import re
import time

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder


class Model_vec:
    def __init__(self, batch_size, dimension_size, learning_rate, vocabulary_size):
        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, dimension_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)
        self.nce_weights = tf.Variable(
            tf.truncated_normal(
                [vocabulary_size, dimension_size], stddev=1.0 / np.sqrt(dimension_size)
            )
        )
        self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=self.nce_weights,
                biases=self.nce_biases,
                labels=self.train_labels,
                inputs=embed,
                num_sampled=batch_size / 2,
                num_classes=vocabulary_size,
            )
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        self.normalized_embeddings = embeddings / self.norm


class Model:
    def __init__(self, dimension_input, size_layer, dimension_output, learning_rate):
        self.X = tf.placeholder(tf.float32, [None, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        layer1 = tf.Variable(tf.random_normal([dimension_input, size_layer], stddev=0.5))
        bias1 = tf.Variable(tf.random_normal([size_layer], stddev=0.1))
        layer2 = tf.Variable(tf.random_normal([size_layer, size_layer], stddev=0.5))
        bias2 = tf.Variable(tf.random_normal([size_layer], stddev=0.1))
        layer3 = tf.Variable(tf.random_normal([size_layer, size_layer], stddev=0.5))
        bias3 = tf.Variable(tf.random_normal([size_layer], stddev=0.1))
        layer4 = tf.Variable(tf.random_normal([size_layer, dimension_output], stddev=0.5))
        bias4 = tf.Variable(tf.random_normal([dimension_output], stddev=0.1))
        feed = tf.nn.tanh(tf.matmul(self.X, layer1) + bias1)
        feed = tf.nn.tanh(tf.matmul(feed, layer2) + bias2)
        feed = tf.nn.tanh(tf.matmul(feed, layer3) + bias3)
        self.logits = tf.matmul(feed, layer4) + bias4
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        l2 = sum(0.0005 * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        self.cost += l2
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


# In[2]:


# In[3]:


def clearstring(string):
    string = re.sub("[^'\"A-Za-z0-9 ]+", "", string)
    string = string.split(" ")
    string = filter(None, string)
    string = [y.strip() for y in string]
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


# In[4]:


def build_dataset(words, vocabulary_size):
    count = []
    count.extend(collections.Counter(words).most_common(vocabulary_size))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary) + 1
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        data.append(index)
    dictionary["PAD"] = 0
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, dictionary, reverse_dictionary


def generate_batch_skipgram(words, batch_size, num_skips, skip_window):
    data_index = 0
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for i in range(span):
        buffer.append(words[data_index])
        data_index = (data_index + 1) % len(words)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(words[data_index])
        data_index = (data_index + 1) % len(words)
    data_index = (data_index + len(words) - span) % len(words)
    return batch, labels


def generatevector(dimension, batch_size, skip_size, skip_window, num_skips, iteration, words_real):

    print("Data size:", len(words_real))
    data, dictionary, reverse_dictionary = build_dataset(words_real, len(words_real))
    sess = tf.InteractiveSession()
    print("Creating Word2Vec model..")
    model = Model_vec(batch_size, dimension, 0.1, len(dictionary))
    sess.run(tf.global_variables_initializer())
    last_time = time.time()
    for step in range(iteration):
        new_time = time.time()
        batch_inputs, batch_labels = generate_batch_skipgram(
            data, batch_size, num_skips, skip_window
        )
        feed_dict = {model.train_inputs: batch_inputs, model.train_labels: batch_labels}
        _, loss = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
        if ((step + 1) % 1000) == 0:
            print(
                "epoch:",
                step + 1,
                ", loss:",
                loss,
                ", speed:",
                (time.time() - new_time) * 1000,
                "s / 1000 epoch",
            )
    tf.reset_default_graph()
    return dictionary, reverse_dictionary, model.normalized_embeddings.eval()


# In[5]:


string, data, label = read_data()
location = os.getcwd()
dimension = 512
skip_size = 8
skip_window = 1
num_skips = 2
iteration_train_vectors = 20000
num_layers = 3
size_layer = 256
learning_rate = 0.0001
epoch = 100
batch = 100
maxlen = 50


# In[6]:


dictionary, reverse_dictionary, vectors = generatevector(
    dimension, 32, skip_size, skip_window, num_skips, iteration_train_vectors, string
)


# In[17]:


train_X, test_X, train_Y, test_Y = train_test_split(data[:, 0], data[:, 1], test_size=0.25)


# In[22]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(dimension, 128, len(label), learning_rate)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 20, 0, 0, 0
batch_size = 200
while True:
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:", EPOCH)
        break
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (train_X.shape[0] // batch) * batch, batch):
        batch_x = np.zeros((batch, dimension))
        batch_y = np.zeros((batch, len(label)))
        for k in range(batch):
            tokens = train_X[i + k].split()
            for no, text in enumerate(tokens):
                try:
                    batch_x[k, :] += vectors[dictionary[text], :]
                except:
                    continue
            batch_y[k, int(train_Y[i + k])] = 1.0
        loss, _ = sess.run(
            [model.cost, model.optimizer], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        train_loss += loss
        train_acc += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})

    for i in range(0, (test_X.shape[0] // batch) * batch, batch):
        batch_x = np.zeros((batch, dimension))
        batch_y = np.zeros((batch, len(label)))
        for k in range(batch):
            tokens = test_X[i + k].split()
            for no, text in enumerate(tokens):
                try:
                    batch_x[k, :] += vectors[dictionary[text], :]
                except:
                    continue
            batch_y[k, int(test_Y[i + k])] = 1.0
        loss, acc = sess.run(
            [model.cost, model.accuracy], feed_dict={model.X: batch_x, model.Y: batch_y}
        )
        test_loss += loss
        test_acc += acc

    train_loss /= train_X.shape[0] // batch
    train_acc /= train_X.shape[0] // batch
    test_loss /= test_X.shape[0] // batch
    test_acc /= test_X.shape[0] // batch
    if test_acc > CURRENT_ACC:
        print("epoch:", EPOCH, ", pass acc:", CURRENT_ACC, ", current acc:", test_acc)
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
        saver.save(sess, os.getcwd() + "/model-rnn-vector.ckpt")
    else:
        CURRENT_CHECKPOINT += 1
    EPOCH += 1
    print(
        "epoch:",
        EPOCH,
        ", training loss:",
        train_loss,
        ", training acc:",
        train_acc,
        ", valid loss:",
        test_loss,
        ", valid acc:",
        test_acc,
    )


# In[ ]:
