#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.datasets
import tensorflow as tf
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import lightgbm as lgb


class Model_vec:
    def __init__(
        self, bow_shape, batch_size, dimension_size, learning_rate, vocabulary_size, boundary
    ):
        self.X = tf.placeholder(tf.float32, shape=[batch_size, bow_shape])
        self.Y = tf.placeholder(tf.int32, shape=[batch_size, 1])
        embeddings = tf.Variable(
            tf.random_uniform([bow_shape, dimension_size], boundary[0], boundary[1])
        )
        embeddings = tf.matmul(self.X, embeddings)
        nce_weights = tf.Variable(
            tf.truncated_normal(
                [vocabulary_size, dimension_size], stddev=1.0 / np.sqrt(dimension_size)
            )
        )
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=self.Y,
                inputs=embeddings,
                num_sampled=batch_size,
                num_classes=vocabulary_size,
            )
        )

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        self.normalized_embeddings = embeddings / norm


# In[2]:


learning_rate = 0.001
boundary = [-1, 1]
batch_size = 20
dimension_size = 300
epoch = 10


# In[3]:


# In[4]:


# clear string
def clearstring(string):
    string = re.sub("[^A-Za-z ]+", "", string)
    string = string.split(" ")
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = " ".join(string)
    return string.lower()


# because of sklean.datasets read a document as a single element
# so we want to split based on new line
def separate_dataset(trainset):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split("\n")
        # python3, if python2, just remove list()
        data_ = list(filter(None, data_))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget


# In[5]:


# you can change any encoding type
trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[6]:


combined = list(zip(trainset.data, trainset.target))
random.shuffle(combined)

trainset.data[:], trainset.target[:] = zip(*combined)


# In[7]:


bow = CountVectorizer(min_df=10).fit(trainset.data)
out = bow.transform(trainset.data)


# In[8]:


label_Y = np.array(trainset.target).reshape((-1, 1))


# In[9]:


sess = tf.InteractiveSession()
model = Model_vec(out.shape[1], batch_size, dimension_size, learning_rate, out.shape[0], boundary)
sess.run(tf.global_variables_initializer())
for i in range(epoch):
    total_loss = 0
    for k in range(0, (out.shape[0] // batch_size) * batch_size, batch_size):
        loss, _ = sess.run(
            [model.loss, model.optimizer],
            feed_dict={
                model.X: out[k : k + batch_size, :].todense(),
                model.Y: label_Y[k : k + batch_size, :],
            },
        )
    total_loss += loss
    print("epoch: ", i, "avg loss: ", total_loss / (out.shape[0] // batch_size))


# In[11]:


vectorized = np.zeros(((out.shape[0] // batch_size) * batch_size, dimension_size))


# In[13]:


for k in range(0, (out.shape[0] // batch_size) * batch_size, batch_size):
    vectorized[k : k + batch_size, :] = sess.run(
        model.normalized_embeddings, feed_dict={model.X: out[k : k + batch_size, :].todense()}
    )


# In[30]:


_, vect_temp, _, Y_temp = train_test_split(
    vectorized, trainset.target[: vectorized.shape[0]], test_size=0.005
)


# In[31]:


embed_2d = TSNE(n_components=2).fit_transform(vect_temp)


# In[32]:


sns.set()
plt.figure(figsize=(10, 10))
colors = sns.color_palette(n_colors=len(trainset.target_names))
y_train_reshape = np.array(Y_temp)
for no, _ in enumerate(np.unique(y_train_reshape)):
    plt.scatter(
        embed_2d[y_train_reshape == no, 0],
        embed_2d[y_train_reshape == no, 1],
        c=colors[no],
        label=trainset.target_names[no],
    )
plt.legend()
plt.show()


# In[34]:


with open("vector.p", "wb") as fopen:
    pickle.dump(vectorized, fopen)
with open("label-y.p", "wb") as fopen:
    pickle.dump(trainset.target[: vectorized.shape[0]], fopen)


# In[35]:


train_X, test_X, train_Y, test_Y = train_test_split(
    vectorized, trainset.target[: vectorized.shape[0]], test_size=0.2
)


# In[36]:


params_lgd = {
    "boosting_type": "dart",
    "objective": "multiclass",
    "colsample_bytree": 0.4,
    "subsample": 0.8,
    "learning_rate": 0.1,
    "silent": False,
    "n_estimators": 10000,
    "reg_lambda": 0.0005,
    "device": "gpu",
}
clf = lgb.LGBMClassifier(**params_lgd)
clf.fit(
    train_X,
    train_Y,
    eval_set=[(train_X, train_Y), (test_X, test_Y)],
    eval_metric="logloss",
    early_stopping_rounds=20,
    verbose=True,
)


# In[37]:


predicted = clf.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[ ]:
