#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# In[2]:


lang = pd.read_csv("sentences.csv", sep="\t")
lang = lang.dropna()
lang.head()


# In[3]:


X, Y = [], []
for no, ln in enumerate(lang.cmn.unique()):
    langs = lang.loc[lang.cmn == ln]
    if langs.shape[0] < 500:
        continue
    print(no, ln)
    langs = langs.iloc[:500, -1].tolist()
    X.extend(langs)
    Y.extend([ln] * len(langs))


# In[4]:


def clean_text(string):
    string = re.sub("[0-9!@#$%^&*()_\-+{}|\~`'\";:?/.>,<]", " ", string.lower(), flags=re.UNICODE)
    return re.sub(r"[ ]+", " ", string.lower()).strip()


# In[5]:


X = [clean_text(s) for s in X]


# In[6]:


bow_chars = CountVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=700000).fit(X)
delattr(bow_chars, "stop_words_")
target = LabelEncoder().fit_transform(Y)
features = bow_chars.transform(X)
features.shape


# In[7]:


train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.2)
del features


# In[8]:


# In[9]:


def convert_sparse_matrix_to_sparse_tensor(X, limit=5):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    coo.data[coo.data > limit] = limit
    return (
        tf.SparseTensorValue(indices, coo.col, coo.shape),
        tf.SparseTensorValue(indices, coo.data, coo.shape),
    )


# In[10]:


labels = np.unique(Y, return_counts=True)[0]
labels


# In[11]:


class Model:
    def __init__(self, learning_rate):
        self.X = tf.sparse_placeholder(tf.int32)
        self.W = tf.sparse_placeholder(tf.int32)
        self.Y = tf.placeholder(tf.int32, [None])
        embeddings = tf.Variable(tf.truncated_normal([train_X.shape[1], 64]))
        embed = tf.nn.embedding_lookup_sparse(embeddings, self.X, self.W, combiner="mean")
        self.logits = tf.layers.dense(embed, len(labels))
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1, output_type=tf.int32), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[12]:


sess = tf.InteractiveSession()
model = Model(1e-4)
sess.run(tf.global_variables_initializer())


# In[13]:


batch_size = 64
for e in range(50):
    lasttime = time.time()
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    pbar = tqdm(range(0, train_X.shape[0], batch_size), desc="train minibatch loop")
    for i in pbar:
        batch_x = convert_sparse_matrix_to_sparse_tensor(
            train_X[i : min(i + batch_size, train_X.shape[0])]
        )
        batch_y = train_Y[i : min(i + batch_size, train_X.shape[0])]
        acc, cost, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X: batch_x[0], model.W: batch_x[1], model.Y: batch_y},
        )
        assert not np.isnan(cost)
        train_loss += cost
        train_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    pbar = tqdm(range(0, test_X.shape[0], batch_size), desc="test minibatch loop")
    for i in pbar:
        batch_x = convert_sparse_matrix_to_sparse_tensor(
            test_X[i : min(i + batch_size, test_X.shape[0])]
        )
        batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
        batch_x_expand = np.expand_dims(batch_x, axis=1)
        acc, cost = sess.run(
            [model.accuracy, model.cost],
            feed_dict={model.X: batch_x[0], model.W: batch_x[1], model.Y: batch_y},
        )
        test_loss += cost
        test_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    train_loss /= train_X.shape[0] / batch_size
    train_acc /= train_X.shape[0] / batch_size
    test_loss /= test_X.shape[0] / batch_size
    test_acc /= test_X.shape[0] / batch_size

    print("time taken:", time.time() - lasttime)
    print(
        "epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n"
        % (e, train_loss, train_acc, test_loss, test_acc)
    )


# In[14]:


real_Y, predict_Y = [], []

pbar = tqdm(range(0, test_X.shape[0], batch_size), desc="validation minibatch loop")
for i in pbar:
    batch_x = convert_sparse_matrix_to_sparse_tensor(
        test_X[i : min(i + batch_size, test_X.shape[0])]
    )
    batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])].tolist()
    predict_Y += np.argmax(
        sess.run(
            model.logits, feed_dict={model.X: batch_x[0], model.W: batch_x[1], model.Y: batch_y}
        ),
        1,
    ).tolist()
    real_Y += batch_y


# In[15]:


print(metrics.classification_report(real_Y, predict_Y, target_names=labels))
