#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re

import numpy as np
import sklearn.datasets
from scipy import sparse
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import (CountVectorizer, HashingVectorizer,
                                             TfidfTransformer, TfidfVectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted, check_X_y


class NB_SVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ["_r", "_clf"])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ["_r", "_clf"])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self


# In[2]:


# In[3]:


def clearstring(string):
    string = re.sub("[^'\"A-Za-z0-9 ]+", "", string)
    string = string.split(" ")
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = " ".join(string)
    return string


def separate_dataset(trainset):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split("\n")
        data_ = list(filter(None, data_))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget


# In[4]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[12]:


cv = StratifiedKFold(n_splits=10, shuffle=True)
X_raw, y_raw = np.array(trainset.data), np.array(trainset.target)

aucs = []
for train, test in cv.split(X_raw, y_raw):
    y_train = y_raw[train]
    X_train = X_raw[train]

    y_test = y_raw[test]
    X_test = X_raw[test]

    tfidf = TfidfVectorizer().fit(X_train)
    X_train_tr = tfidf.transform(X_train)
    X_test_tr = tfidf.transform(X_test)

    model = NB_SVM(C=1, dual=True, n_jobs=-1).fit(X_train_tr, y_train)
    y_preds = model.predict(X_test_tr)
    accuracy = accuracy_score(y_test, y_preds)
    aucs.append(accuracy)
    print(accuracy)


# In[5]:


# bag-of-word
bow = CountVectorizer().fit_transform(trainset.data)

# tf-idf, must get from BOW first
tfidf = TfidfTransformer().fit_transform(bow)

# hashing, default n_features, probability cannot divide by negative
hashing = HashingVectorizer(non_negative=True).fit_transform(trainset.data)


# In[7]:


train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size=0.2)

model = NB_SVM(C=1, dual=True, n_jobs=-1).fit(train_X, train_Y)

predicted = model.predict(test_X)

print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[8]:


train_X, test_X, train_Y, test_Y = train_test_split(tfidf, trainset.target, test_size=0.2)

model = NB_SVM(C=1, dual=True, n_jobs=-1).fit(train_X, train_Y)

predicted = model.predict(test_X)

print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[9]:


train_X, test_X, train_Y, test_Y = train_test_split(hashing, trainset.target, test_size=0.2)

model = NB_SVM(C=1, dual=True, n_jobs=-1).fit(train_X, train_Y)

predicted = model.predict(test_X)

print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[ ]:
