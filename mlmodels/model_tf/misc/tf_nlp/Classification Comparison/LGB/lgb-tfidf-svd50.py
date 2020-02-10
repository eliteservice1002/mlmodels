#!/usr/bin/env python
# coding: utf-8

# # GPU Light gradient boosting trained on TF-IDF reduced 50 dimensions
#
# 1. Same emotion dataset from [NLP-dataset](https://github.com/huseinzol05/NLP-Dataset)
# 2. Same splitting 80% training, 20% testing, may vary depends on randomness
# 3. Same regex substitution '[^\"\'A-Za-z0-9 ]+'
#
# ## Example
#
# Based on Term-frequency Inverse document frequency
#
# After that we apply SVD to reduce the dimensions, n_components = 50

# In[8]:


import re
import time

import numpy as np
import sklearn.datasets
from sklearn import metrics, pipeline
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

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


train_X, test_X, train_Y, test_Y = train_test_split(
    trainset_data.data, trainset_data.target, test_size=0.2
)


# In[5]:


decompose = pipeline.Pipeline(
    [("count", TfidfVectorizer()), ("svd", TruncatedSVD(n_components=50))]
).fit(trainset_data.data)


# In[6]:


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


# In[10]:


train_X = decompose.transform(train_X)
test_X = decompose.transform(test_X)


# In[11]:


d_train = lgb.Dataset(train_X, train_Y)
d_valid = lgb.Dataset(test_X, test_Y)
watchlist = [d_train, d_valid]
t = time.time()
clf = lgb.train(params_lgb, d_train, 100000, watchlist, early_stopping_rounds=200, verbose_eval=100)
print(round(time.time() - t, 3), "Seconds to train lgb")


# In[12]:


print(
    metrics.classification_report(
        test_Y, np.argmax(clf.predict(test_X), axis=1), target_names=trainset_data.target_names
    )
)


# In[13]:


clf.save_model("lgb-tfidf-svd50.model")


# In[ ]:
