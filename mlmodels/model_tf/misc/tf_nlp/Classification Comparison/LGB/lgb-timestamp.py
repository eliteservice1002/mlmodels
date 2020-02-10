#!/usr/bin/env python
# coding: utf-8

# # GPU Light Gradient boosting trained on timestamp text data-set
#
# 1. Same emotion dataset from [NLP-dataset](https://github.com/huseinzol05/NLP-Dataset)
# 2. Same splitting 80% training, 20% testing, may vary depends on randomness
# 3. Same regex substitution '[^\"\'A-Za-z0-9 ]+'
#
# ## Example
#
# Based on sorted dictionary position
#
# text: 'module into which all the refactored classes', matrix: [167, 143, 12, 3, 4, 90]

# In[1]:


import json
import pickle
import re
import time

import numpy as np
import sklearn.datasets
from sklearn import metrics
from sklearn.cross_validation import train_test_split

import lightgbm as lgb

# In[2]:


def clearstring(string):
    string = re.sub("[^\"'A-Za-z0-9 ]+", "", string)
    string = string.split(" ")
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = " ".join(string)
    return string


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


# In[3]:


trainset_data = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset_data.data, trainset_data.target = separate_dataset(trainset_data)


# In[4]:


with open("dictionary_emotion.p", "rb") as fopen:
    dict_emotion = pickle.load(fopen)


# In[5]:


len_sentences = np.array([len(i.split()) for i in trainset_data.data])
maxlen = np.ceil(len_sentences.mean()).astype("int")
data_X = np.zeros((len(trainset_data.data), maxlen))


# In[6]:


for i in range(data_X.shape[0]):
    tokens = trainset_data.data[i].split()[:maxlen]
    for no, text in enumerate(tokens[::-1]):
        try:
            data_X[i, -1 - no] = dict_emotion[text]
        except:
            continue


# In[7]:


train_X, test_X, train_Y, test_Y = train_test_split(data_X, trainset_data.target, test_size=0.2)


# In[8]:


params_lgb = {
    "max_depth": 27,
    "learning_rate": 0.03,
    "verbose": 50,
    "early_stopping_round": 200,
    "metric": "multi_logloss",
    "objective": "multiclass",
    "num_classes": len(trainset_data.target_names),
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
}


# In[12]:


d_train = lgb.Dataset(train_X, train_Y)
d_valid = lgb.Dataset(test_X, test_Y)
watchlist = [d_train, d_valid]
t = time.time()
clf = lgb.train(params_lgb, d_train, 100000, watchlist, early_stopping_rounds=200, verbose_eval=100)
print(round(time.time() - t, 3), "Seconds to train lgb")


# In[14]:


np.mean(test_Y == np.argmax(clf.predict(test_X), axis=1))


# In[15]:


clf.save_model("lgb-timestamp.model")


# In[16]:


print(
    metrics.classification_report(
        test_Y, np.argmax(clf.predict(test_X), axis=1), target_names=trainset_data.target_names
    )
)


# In[ ]:
