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


maxlen = 100
kernel_size = 3
batch_size = 32
embedded_size = 128


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


X = tf.keras.preprocessing.sequence.pad_sequences(idx_trainset, maxlen)
X.shape


# In[7]:


train_X, test_X, train_Y, test_Y = train_test_split(X, trainset.target, test_size=0.2)


# In[8]:


def gated_linear_unit(x, d_rate):
    c = tf.layers.conv1d(
        inputs=x,
        filters=2 * embedded_size,
        kernel_size=kernel_size,
        dilation_rate=d_rate,
        padding="same",
    )
    c1, c2 = tf.split(c, 2, -1)
    x += c1 * tf.sigmoid(c2)
    return x


class Model:
    def __init__(self, embedded_size, dict_size, dimension_output, learning_rate):

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)

        for d_rate in [1, 2, 4]:
            encoder_embedded = gated_linear_unit(encoder_embedded, d_rate)

        encoder_embedded = tf.reduce_max(encoder_embedded, 1)
        encoder_embedded = tf.layers.flatten(encoder_embedded)
        forward = tf.layers.dense(encoder_embedded, embedded_size, tf.nn.relu)
        self.logits = tf.layers.dense(forward, dimension_output)
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1, output_type=tf.int32), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[9]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(embedded_size, vocabulary_size + 4, 2, 1e-3)
sess.run(tf.global_variables_initializer())


# In[10]:


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


# In[11]:


logits = sess.run(model.logits, feed_dict={model.X: test_X})
print(
    metrics.classification_report(test_Y, np.argmax(logits, 1), target_names=trainset.target_names)
)


# In[ ]:
