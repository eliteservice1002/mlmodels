#!/usr/bin/env python
# coding: utf-8

# Here I will to show how to use linear model stochastic gradient descent on multi-class classification/discrimination
#
# import class sklearn.linear_model.SGDClassifier

# In[1]:


import re

import numpy as np
import sklearn.datasets
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

import xgboost as xgb

# In[2]:


xgb.__version__


# Define some functions to help us on preprocessing

# In[3]:


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


# In[4]:


# you can change any encoding type
trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[5]:


bow = TfidfVectorizer(min_df=10).fit(trainset.data)


# In[6]:


out = bow.transform(trainset.data)


# In[7]:


trainset.target = np.array(trainset.target)
train_X, test_X, train_Y, test_Y = train_test_split(out, trainset.target, test_size=0.2)


# In[8]:


params_xgd = {
    "min_child_weight": 10.0,
    "max_depth": 7,
    "objective": "multi:softprob",
    "max_delta_step": 1.8,
    "colsample_bytree": 0.4,
    "subsample": 0.8,
    "learning_rate": 0.1,
    "gamma": 0.65,
    "nthread": -1,
    "silent": False,
    "n_estimators": 10000,
}
clf = xgb.XGBClassifier(**params_xgd)
clf.fit(
    train_X,
    train_Y,
    eval_set=[(train_X, train_Y), (test_X, test_Y)],
    eval_metric="mlogloss",
    early_stopping_rounds=20,
    verbose=True,
)


# In[9]:


predicted = clf.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))
