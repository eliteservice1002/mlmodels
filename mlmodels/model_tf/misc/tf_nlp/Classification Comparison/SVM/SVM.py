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
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

# Define some functions to help us on preprocessing

# In[2]:


# clear string
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


# you can change any encoding type
trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[4]:


# bag-of-word
bow = CountVectorizer().fit_transform(trainset.data)

# tf-idf, must get from BOW first
tfidf = TfidfTransformer().fit_transform(bow)

# hashing, default n_features, probability cannot divide by negative
hashing = HashingVectorizer(non_negative=True).fit_transform(trainset.data)


# #### loss function got {'modified_huber', 'hinge', 'log', 'squared_hinge', 'perceptron'}
#
# default is hinge, will give you classic SVM
#
# perceptron in linear loss
#
# huber and log both logistic classifier
#
# #### penalty got {'l1', 'l2'}, to prevent overfitting
#
# l1 = MAE (mean absolute error)
#
# l2 = RMSE (root mean square error)
#
# #### alpha is learning rate
#
# #### n_iter is number of epoch

# In[5]:


train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size=0.2)

mod_huber = SGDClassifier(loss="modified_huber", penalty="l2", alpha=1e-3, n_iter=10).fit(
    train_X, train_Y
)
predicted = mod_huber.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[6]:


train_X, test_X, train_Y, test_Y = train_test_split(tfidf, trainset.target, test_size=0.2)

mod_huber = SGDClassifier(loss="modified_huber", penalty="l2", alpha=1e-3, n_iter=10).fit(
    train_X, train_Y
)
predicted = mod_huber.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[7]:


train_X, test_X, train_Y, test_Y = train_test_split(hashing, trainset.target, test_size=0.2)

mod_huber = SGDClassifier(loss="modified_huber", penalty="l2", alpha=1e-3, n_iter=10).fit(
    train_X, train_Y
)
predicted = mod_huber.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# Always BOW got the highest accuracy among other vectorization

# Now let we use linear model to do classifers, I will use BOW as vectorizer

# In[8]:


train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size=0.2)

svm = SGDClassifier(penalty="l2", alpha=1e-3, n_iter=10).fit(train_X, train_Y)
predicted = svm.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[9]:


train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size=0.2)

sq_hinge = SGDClassifier(loss="squared_hinge", penalty="l2", alpha=1e-3, n_iter=10).fit(
    train_X, train_Y
)
predicted = sq_hinge.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[10]:


train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size=0.2)

perceptron = SGDClassifier(loss="perceptron", penalty="l2", alpha=1e-3, n_iter=10).fit(
    train_X, train_Y
)
predicted = perceptron.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# But how to get probability of our output?
#
# Only applicable if your loss = {'log', 'modified_huber'} because both are logistic regression

# In[11]:


train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size=0.2)

mod_huber = SGDClassifier(loss="modified_huber", penalty="l2", alpha=1e-3, n_iter=10).fit(
    train_X, train_Y
)
predicted = mod_huber.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))

# get probability for first 2 sentence in our dataset
print(trainset.data[:2])
print(trainset.target[:2])
print(mod_huber.predict_proba(bow[:2, :]))
