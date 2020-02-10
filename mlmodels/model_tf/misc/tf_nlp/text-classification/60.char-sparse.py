#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import time

import tensorflow as tf
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from utils import *

# In[2]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[4]:


def convert_sparse_matrix_to_sparse_tensor(X, limit=5):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    coo.data[coo.data > limit] = limit
    return (
        tf.SparseTensorValue(indices, coo.col, coo.shape),
        tf.SparseTensorValue(indices, coo.data, coo.shape),
    )


# In[5]:


bow_chars = CountVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=300000).fit(
    trainset.data
)
delattr(bow_chars, "stop_words_")


# In[6]:


feature_shape = bow_chars.transform(trainset.data[:1]).shape[1]
feature_shape


# In[7]:


class Model:
    def __init__(self, output_size, vocab_size, learning_rate):
        self.X = tf.sparse_placeholder(tf.int32)
        self.W = tf.sparse_placeholder(tf.int32)
        self.Y = tf.placeholder(tf.int32, [None])
        embeddings = tf.Variable(tf.truncated_normal([vocab_size, 64]))
        embed = tf.nn.embedding_lookup_sparse(embeddings, self.X, self.W, combiner="mean")
        self.logits = tf.layers.dense(embed, output_size)
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1, output_type=tf.int32), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[8]:


sess = tf.InteractiveSession()
model = Model(len(trainset.target_names), feature_shape, 1e-4)
sess.run(tf.global_variables_initializer())


# In[9]:


vectors = bow_chars.transform(trainset.data)
train_X, test_X, train_Y, test_Y = train_test_split(vectors, trainset.target, test_size=0.2)


# In[10]:


batch_size = 32
EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 3, 0, 0, 0

while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print("break epoch:%d\n" % (EPOCH))
        break

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    pbar = tqdm(range(0, train_X.shape[0], batch_size), desc="train minibatch loop")
    for i in pbar:
        batch_x = convert_sparse_matrix_to_sparse_tensor(
            train_X[i : min(i + batch_size, train_X.shape[0])]
        )
        batch_y = train_Y[i : min(i + batch_size, train_X.shape[0])]
        batch_x_expand = np.expand_dims(batch_x, axis=1)
        acc, cost, _ = sess.run(
            [model.accuracy, model.cost, model.optimizer],
            feed_dict={model.Y: batch_y, model.X: batch_x[0], model.W: batch_x[1]},
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
            feed_dict={model.Y: batch_y, model.X: batch_x[0], model.W: batch_x[1]},
        )
        test_loss += cost
        test_acc += acc
        pbar.set_postfix(cost=cost, accuracy=acc)

    train_loss /= train_X.shape[0] / batch_size
    train_acc /= train_X.shape[0] / batch_size
    test_loss /= test_X.shape[0] / batch_size
    test_acc /= test_X.shape[0] / batch_size

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


real_Y, predict_Y = [], []

pbar = tqdm(range(0, test_X.shape[0], batch_size), desc="validation minibatch loop")
for i in pbar:
    batch_x = convert_sparse_matrix_to_sparse_tensor(
        test_X[i : min(i + batch_size, test_X.shape[0])]
    )
    batch_y = test_Y[i : min(i + batch_size, test_X.shape[0])]
    predict_Y += np.argmax(
        sess.run(
            model.logits, feed_dict={model.X: batch_x[0], model.W: batch_x[1], model.Y: batch_y}
        ),
        1,
    ).tolist()
    real_Y += batch_y


# In[12]:


print(metrics.classification_report(real_Y, predict_Y, target_names=["negative", "positive"]))


# In[ ]:
