#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import time

import tensorflow as tf
from sklearn.cross_validation import train_test_split
from tqdm import tqdm

import xl
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


learning_rate = 1e-4
batch_size = 32
maxlen = 100
n_layer = 3
d_model = 256
d_embed = 256
n_head = 10
d_head = 50
d_inner = 512


# In[6]:


class Model:
    def __init__(self):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None])
        self.memory = tf.fill([n_layer, tf.shape(self.X)[0], tf.shape(self.X)[1], d_model], PAD)
        self.memory = tf.cast(self.memory, tf.float32)
        initializer = tf.initializers.random_normal(stddev=0.1)
        logits, self.next_memory = xl.transformer(
            self.X,
            self.memory,
            len(dictionary),
            n_layer,
            d_model,
            d_embed,
            n_head,
            d_head,
            d_inner,
            initializer,
        )
        logits = tf.reduce_mean(logits, axis=1)
        self.logits = tf.layers.dense(logits, 2)
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1, output_type=tf.int32), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[7]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())


# In[8]:


vectors = str_idx(trainset.data, dictionary, maxlen)
train_X, test_X, train_Y, test_Y = train_test_split(vectors, trainset.target, test_size=0.2)


# In[9]:


EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 3, 0, 0, 0

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
        acc, cost, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.Y: batch_y, model.X: batch_x},
        )
        assert not np.isnan(cost)
        train_loss += cost
        train_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    pbar = tqdm(range(0, len(test_X), batch_size), desc="test minibatch loop")
    for i in pbar:
        batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
        batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
        acc, cost = sess.run(
            [model.accuracy, model.cost], feed_dict={model.Y: batch_y, model.X: batch_x}
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
    batch_memory = np.ones(([n_layer, batch_size, maxlen, d_model]))
    predict_Y += np.argmax(
        sess.run(
            model.logits, feed_dict={model.X: batch_x, model.Y: batch_y, model.memory: batch_memory}
        ),
        1,
    ).tolist()
    real_Y += batch_y


# In[11]:


print(metrics.classification_report(real_Y, predict_Y, target_names=trainset.target_names))


# In[12]:


batch_x.shape


# In[ ]:
