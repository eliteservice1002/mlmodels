#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import time

import tensorflow as tf
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from tqdm import tqdm

from utils import *

# In[2]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[3]:


concat = " ".join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print("vocab from size: %d" % (vocabulary_size))
print("Most common words", count[4:10])
print("Sample data", data[:10], [rev_dictionary[i] for i in data[:10]])


# In[4]:


GO = dictionary["GO"]
PAD = dictionary["PAD"]
EOS = dictionary["EOS"]
UNK = dictionary["UNK"]


# In[5]:


size_layer = 128
dimension_output = len(trainset.target_names)
maxlen = 50
batch_size = 128


# In[6]:


def hop_forward(memory_o, memory_i, response_proj, inputs_len, questions_len):
    match = memory_i
    match = pre_softmax_masking(match, inputs_len)
    match = tf.nn.softmax(match)
    match = post_softmax_masking(match, questions_len)
    response = tf.multiply(match, memory_o)
    return response_proj(response)


def pre_softmax_masking(x, seq_len):
    paddings = tf.fill(tf.shape(x), float("-inf"))
    T = tf.shape(x)[1]
    max_seq_len = tf.shape(x)[2]
    masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
    masks = tf.tile(tf.expand_dims(masks, 1), [1, T, 1])
    return tf.where(tf.equal(masks, 0), paddings, x)


def post_softmax_masking(x, seq_len):
    T = tf.shape(x)[2]
    max_seq_len = tf.shape(x)[1]
    masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
    masks = tf.tile(tf.expand_dims(masks, -1), [1, 1, T])
    return x * masks


def shift_right(x):
    batch_size = tf.shape(x)[0]
    start = tf.to_int32(tf.fill([batch_size, 1], GO))
    return tf.concat([start, x[:, :-1]], 1)


def embed_seq(x, vocab_size, zero_pad=True):
    lookup_table = tf.get_variable("lookup_table", [vocab_size, size_layer], tf.float32)
    if zero_pad:
        lookup_table = tf.concat((tf.zeros([1, size_layer]), lookup_table[1:, :]), axis=0)
    return tf.nn.embedding_lookup(lookup_table, x)


def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


def quest_mem(x, vocab_size, max_quest_len):
    x = embed_seq(x, vocab_size)
    pos = position_encoding(max_quest_len, size_layer)
    return x * pos


class QA:
    def __init__(self, vocab_size, size_layer, learning_rate, dimension_output, n_hops=3):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])

        lookup_table = tf.get_variable("lookup_table", [vocab_size, size_layer], tf.float32)

        with tf.variable_scope("memory_o"):
            memory_o = quest_mem(self.X, vocab_size, maxlen)

        with tf.variable_scope("memory_i"):
            memory_i = quest_mem(self.X, vocab_size, maxlen)

        with tf.variable_scope("interaction"):
            response_proj = tf.layers.Dense(size_layer)
            for _ in range(n_hops):
                answer = hop_forward(
                    memory_o, memory_i, response_proj, self.X_seq_len, self.X_seq_len
                )
                memory_i = answer
        W = tf.get_variable(
            "w", shape=(size_layer, dimension_output), initializer=tf.orthogonal_initializer()
        )
        b = tf.get_variable("b", shape=(dimension_output), initializer=tf.zeros_initializer())
        self.logits = tf.matmul(answer[:, -1], W) + b
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.predictions = tf.argmax(self.logits, 1)
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[7]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = QA(len(dictionary), size_layer, 1e-2, dimension_output)
sess.run(tf.global_variables_initializer())


# In[8]:


vectors = str_idx(trainset.data, dictionary, maxlen)
train_X, test_X, train_Y, test_Y = train_test_split(vectors, trainset.target, test_size=0.2)


# In[9]:


EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0

while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:%d\n" % (EPOCH))
        break

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    pbar = tqdm(range(0, len(train_X), batch_size), desc="train minibatch loop")
    for i in pbar:
        batch_x = train_X[i : min(i + batch_size, train_X.shape[0])]
        batch_y = train_Y[i : min(i + batch_size, train_X.shape[0])]
        batch_x_len = [maxlen] * len(batch_x)
        acc, cost, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.Y: batch_y, model.X: batch_x, model.X_seq_len: batch_x_len},
        )
        assert not np.isnan(cost)
        train_loss += cost
        train_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    pbar = tqdm(range(0, len(test_X), batch_size), desc="test minibatch loop")
    for i in pbar:
        batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
        batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
        batch_x_len = [maxlen] * len(batch_x)
        acc, cost = sess.run(
            [model.accuracy, model.cost],
            feed_dict={model.Y: batch_y, model.X: batch_x, model.X_seq_len: batch_x_len},
        )
        test_loss += cost
        test_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    train_loss /= len(train_X) / batch_size
    train_acc /= len(train_X) / batch_size
    test_loss /= len(test_X) / batch_size
    test_acc /= len(test_X) / batch_size

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


# In[10]:


real_Y, predict_Y = [], []

pbar = tqdm(range(0, len(test_X), batch_size), desc="validation minibatch loop")
for i in pbar:
    batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
    batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
    batch_x_len = [maxlen] * len(batch_x)
    predict_Y += np.argmax(
        sess.run(
            model.logits,
            feed_dict={model.Y: batch_y, model.X: batch_x, model.X_seq_len: batch_x_len},
        ),
        1,
    ).tolist()
    real_Y += batch_y


# In[11]:


print(metrics.classification_report(real_Y, predict_Y, target_names=["negative", "positive"]))


# In[ ]:
