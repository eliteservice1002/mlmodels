#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re

import numpy as np
import sklearn.datasets
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

import lightgbm as lgb

# In[2]:


lgb.__version__


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


tfidf = TfidfVectorizer(min_df=10).fit(trainset.data)


# In[6]:


out = tfidf.transform(trainset.data)


# In[7]:


trainset.target = np.array(trainset.target)
train_X, test_X, train_Y, test_Y = train_test_split(out, trainset.target, test_size=0.2)


# In[11]:


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


# In[12]:


predicted = clf.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[ ]:
