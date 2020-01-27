#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import time

import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.cross_validation import train_test_split

from utils import *

# In[2]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[3]:


train_X, test_X, train_Y, test_Y = train_test_split(trainset.data, trainset.target, test_size=0.2)


# In[4]:


concat = " ".join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print("vocab from size: %d" % (vocabulary_size))
print("Most common words", count[4:10])
print("Sample data", data[:10], [rev_dictionary[i] for i in data[:10]])


# In[5]:


GO = dictionary["GO"]
PAD = dictionary["PAD"]
EOS = dictionary["EOS"]
UNK = dictionary["UNK"]


# In[6]:


class Model:
    def __init__(
        self, size_layer, num_layers, embedded_size, dict_size, dimension_output, margin=0.2
    ):
        def cells(reuse=False):
            return tf.nn.rnn_cell.BasicRNNCell(size_layer, reuse=reuse)

        def rnn(embedded, reuse=False):
            with tf.variable_scope("model", reuse=reuse):
                rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
                outputs, _ = tf.nn.dynamic_rnn(rnn_cells, embedded, dtype=tf.float32)
                W = tf.get_variable(
                    "w",
                    shape=(size_layer, dimension_output),
                    initializer=tf.orthogonal_initializer(),
                )
                b = tf.get_variable(
                    "b", shape=(dimension_output), initializer=tf.zeros_initializer()
                )
                return tf.matmul(outputs[:, -1], W) + b

        with tf.device("/cpu:0"):
            self.INPUT_1 = tf.placeholder(tf.int32, [None, None])
            self.INPUT_2 = tf.placeholder(tf.int32, [None, None])
            self.Y = tf.placeholder(tf.float32, [None, 1])
            encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
            input1_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.INPUT_1)
            input2_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.INPUT_2)
            self.logits_1 = rnn(input1_embedded, False)
            self.logits_2 = rnn(input2_embedded, True)
            d = tf.sqrt(tf.reduce_sum(tf.pow(self.logits_1 - self.logits_2, 2), 1, keep_dims=True))
            tmp = self.Y * tf.square(d)
            tmp2 = (1 - self.Y) * tf.square(tf.maximum((margin - d), 0))
            self.cost = tf.reduce_mean(tmp + tmp2) / 2
            self.optimizer = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(
                self.cost
            )


# In[7]:


size_layer = 128
num_layers = 2
embedded_size = 128
dimension_output = 32
maxlen = 50
batch_size = 128


# In[8]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(size_layer, num_layers, embedded_size, vocabulary_size + 4, dimension_output)
sess.run(tf.global_variables_initializer())


# In[9]:


c = list(zip(train_X, train_Y))
random.shuffle(c)
train_X_1, train_Y_1 = zip(*c)

c = list(zip(train_X, train_Y))
random.shuffle(c)
train_X_2, train_Y_2 = zip(*c)

label_shuffle = np.expand_dims((np.array(train_Y_1) == np.array(train_Y_2)).astype("int"), 1)


# In[10]:


for i in range(50):
    total_loss = 0
    lasttime = time.time()
    for k in range(0, (len(train_X) // batch_size) * batch_size, batch_size):
        batch_x_1 = str_idx(train_X_1[i : i + batch_size], dictionary, maxlen)
        batch_x_2 = str_idx(train_X_2[i : i + batch_size], dictionary, maxlen)
        batch_y = label_shuffle[i : i + batch_size]
        loss, _ = sess.run(
            [model.cost, model.optimizer],
            feed_dict={model.INPUT_1: batch_x_1, model.INPUT_2: batch_x_2, model.Y: batch_y},
        )
        total_loss += loss
    total_loss /= len(train_X) // batch_size
    print("time taken:", time.time() - lasttime)
    print("epoch: %d, training loss: %f\n" % (i, total_loss))


# In[11]:


batch_x = str_idx(train_X_1, dictionary, maxlen)
batch_y = str_idx(test_X, dictionary, maxlen)


# In[12]:


logits_train = sess.run(model.logits_1, feed_dict={model.INPUT_1: batch_x})
logits_test = sess.run(model.logits_1, feed_dict={model.INPUT_1: batch_y})


# In[26]:


label_test = []
for i in range(logits_test.shape[0]):
    label_test.append(
        train_Y_1[np.argsort(cdist(logits_train, [logits_test[i, :]], "cosine").ravel())[0]]
    )


# In[28]:


print(metrics.classification_report(test_Y, label_test, target_names=trainset.target_names))


# In[ ]:
