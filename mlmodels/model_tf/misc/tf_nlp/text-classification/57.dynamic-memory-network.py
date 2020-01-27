#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from tqdm import tqdm

from dynamic_memory_network import DynamicMemoryNetwork
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
num_layers = 2
embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 128
decay_step = 1e4
decay_rate = 1.0
story_len = 1


# In[6]:


tf.reset_default_graph()
sess = tf.InteractiveSession()

model = DynamicMemoryNetwork(
    dimension_output,
    learning_rate,
    decay_step,
    decay_rate,
    maxlen,
    story_len,
    len(dictionary),
    embedded_size,
    size_layer,
)

sess.run(tf.global_variables_initializer())


# In[7]:


vectors = str_idx(trainset.data, dictionary, maxlen)
train_X, test_X, train_Y, test_Y = train_test_split(vectors, trainset.target, test_size=0.2)


# In[8]:


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
        batch_x_expand = np.expand_dims(batch_x, axis=1)
        acc, cost, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={
                model.answer_single: batch_y,
                model.query: batch_x,
                model.story: batch_x_expand,
                model.dropout_keep_prob: 1.0,
            },
        )
        assert not np.isnan(cost)
        train_loss += cost
        train_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    pbar = tqdm(range(0, len(test_X), batch_size), desc="test minibatch loop")
    for i in pbar:
        batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
        batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
        batch_x_expand = np.expand_dims(batch_x, axis=1)
        acc, cost = sess.run(
            [model.accuracy, model.cost],
            feed_dict={
                model.answer_single: batch_y,
                model.query: batch_x,
                model.story: batch_x_expand,
                model.dropout_keep_prob: 1.0,
            },
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


# In[9]:


real_Y, predict_Y = [], []

pbar = tqdm(range(0, len(test_X), batch_size), desc="validation minibatch loop")
for i in pbar:
    batch_x = test_X[i : min(i + batch_size, test_X.shape[0])]
    batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
    batch_x_expand = np.expand_dims(batch_x, axis=1)
    predict_Y += np.argmax(
        sess.run(
            model.logits,
            feed_dict={
                model.answer_single: batch_y,
                model.query: batch_x,
                model.story: batch_x_expand,
                model.dropout_keep_prob: 1.0,
            },
        ),
        1,
    ).tolist()
    real_Y += batch_y


# In[10]:


print(metrics.classification_report(real_Y, predict_Y, target_names=["negative", "positive"]))


# In[ ]:
