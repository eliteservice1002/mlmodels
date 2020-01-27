#!/usr/bin/env python
# coding: utf-8

# # GPU Extreme gradient boosting trained on TF-IDF reduced 50 dimensions
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

# In[1]:


import json
import re

import numpy as np
import sklearn.datasets
from sklearn import metrics, pipeline
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

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


params_xgd = {
    "min_child_weight": 10.0,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": len(trainset_data.target_names),
    "max_depth": 7,
    "max_delta_step": 1.8,
    "colsample_bytree": 0.4,
    "subsample": 0.8,
    "eta": 0.03,
    "gamma": 0.65,
    "num_boost_round": 700,
    "gpu_id": 0,
    "tree_method": "gpu_hist",
}


# In[7]:


train_X = decompose.transform(train_X)
test_X = decompose.transform(test_X)


# In[8]:


d_train = xgb.DMatrix(train_X, train_Y)
d_valid = xgb.DMatrix(test_X, test_Y)
watchlist = [(d_train, "train"), (d_valid, "valid")]
# with open('clf.p', 'rb') as fopen:
#    clf = pickle.load(fopen)
clf = xgb.train(
    params_xgd,
    d_train,
    100000,
    watchlist,
    early_stopping_rounds=100,
    maximize=False,
    verbose_eval=50,
)


# In[9]:


np.mean(
    test_Y == np.argmax(clf.predict(xgb.DMatrix(test_X), ntree_limit=clf.best_ntree_limit), axis=1)
)


# In[10]:


print(
    metrics.classification_report(
        test_Y,
        np.argmax(clf.predict(xgb.DMatrix(test_X), ntree_limit=clf.best_ntree_limit), axis=1),
        target_names=trainset_data.target_names,
    )
)


# In[12]:


clf.save_model("xgb-tfidf-svd50.model")
bst = xgb.Booster(params_xgd)
bst.load_model("xgb-tfidf-svd50.model")

with open("xgb-tfidf-svd50-param", "w") as fopen:
    fopen.write(json.dumps(params_xgd))
np.mean(test_Y == np.argmax(bst.predict(xgb.DMatrix(test_X)), axis=1))


# In[ ]:
