#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import time

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from tqdm import tqdm

from utils import *

# In[2]:


ngram_range = 2
max_features = 20000
maxlen = 50
batch_size = 64
embedded_size = 128
epoch = 10


# In[3]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[4]:


concat = " ".join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print("vocab from size: %d" % (vocabulary_size))
print("Most common words", count[4:10])
print("Sample data", data[:10], [rev_dictionary[i] for i in data[:10]])


# In[5]:


idx_trainset = []
for text in trainset.data:
    idx = []
    for t in text.split():
        try:
            idx.append(dictionary[t])
        except:
            pass
    idx_trainset.append(idx)


# In[6]:


def create_ngram_set(input_list, ngram_value):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def build_ngram(x_train):
    global max_features
    ngram_set = set()
    for input_list in tqdm(x_train, total=len(x_train), ncols=70):
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    max_features = np.max(list(indice_token.keys())) + 1
    return token_indice


def add_ngram(sequences, token_indice):
    new_sequences = []
    for input_list in tqdm(sequences, total=len(sequences), ncols=70):
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i : i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences


# In[7]:


token_indice = build_ngram(idx_trainset)
X = add_ngram(idx_trainset, token_indice)
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen)


# In[8]:


X.shape


# In[9]:


train_X, test_X, train_Y, test_Y = train_test_split(X, trainset.target, test_size=0.2)


# In[10]:


class Model:
    def __init__(self, embedded_size, dict_size, dimension_output, learning_rate):

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        self.logits = tf.layers.dense(tf.reduce_mean(encoder_embedded, 1), dimension_output)
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1, output_type=tf.int32), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[11]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(embedded_size, vocabulary_size + 4, 2, 1e-3)
sess.run(tf.global_variables_initializer())


# In[12]:


EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:%d\n" % (EPOCH))
        break

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (len(train_X) // batch_size) * batch_size, batch_size):
        acc, loss, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X: train_X[i : i + batch_size], model.Y: train_Y[i : i + batch_size]},
        )
        train_loss += loss
        train_acc += acc

    for i in range(0, (len(test_X) // batch_size) * batch_size, batch_size):
        acc, loss = sess.run(
            [model.accuracy, model.cost],
            feed_dict={model.X: test_X[i : i + batch_size], model.Y: test_Y[i : i + batch_size]},
        )
        test_loss += loss
        test_acc += acc

    train_loss /= len(train_X) // batch_size
    train_acc /= len(train_X) // batch_size
    test_loss /= len(test_X) // batch_size
    test_acc /= len(test_X) // batch_size

    if test_acc > CURRENT_ACC:
        print("epoch: %d, pass acc: %f, current acc: %f" % (EPOCH, CURRENT_ACC, test_acc))
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
    else:
        CURRENT_CHECKPOINT += 1

    print("time taken:", time.time() - lasttime)
    print(
        "epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n"
        % (EPOCH, train_loss, train_acc, test_loss, test_acc)
    )
    EPOCH += 1


# In[13]:


logits = sess.run(model.logits, feed_dict={model.X: test_X})
print(
    metrics.classification_report(test_Y, np.argmax(logits, 1), target_names=trainset.target_names)
)


# In[ ]:
