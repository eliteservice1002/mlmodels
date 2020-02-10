#!/usr/bin/env python
# coding: utf-8

# Here I will to show how to use bayes on multi-class classification/discrimination
#
# import class sklearn.naive_bayes.MultinomialNB for Multinomial logistic regression (logistic regression of multi-class)
#
# But if you want to classify binary/boolean class, it is better to use BernoulliNB

# I will use also compare accuracy for using BOW, TF-IDF, and HASHING for vectorizing technique

# In[1]:


import re

import numpy as np
import sklearn.datasets
# to get f1 score
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# Define some function to help us for preprocessing

# In[2]:


# clear string
def clearstring(string):
    string = re.sub("[^'\"A-Za-z0-9 ]+", "", string)
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


# Feed Naive Bayes using BOW
#
# but split it first into train-set (80% of our data-set), and validation-set (20% of our data-set)

# In[5]:


train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size=0.2)

bayes_multinomial = MultinomialNB().fit(train_X, train_Y)
predicted = bayes_multinomial.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# Feed Naive Bayes using TF-IDF
#
# but split it first into train-set (80% of our data-set), and validation-set (20% of our data-set)

# In[6]:


train_X, test_X, train_Y, test_Y = train_test_split(tfidf, trainset.target, test_size=0.2)

bayes_multinomial = MultinomialNB().fit(train_X, train_Y)
predicted = bayes_multinomial.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# Feed Naive Bayes using hashing
#
# but split it first into train-set (80% of our data-set), and validation-set (20% of our data-set)

# In[7]:


train_X, test_X, train_Y, test_Y = train_test_split(hashing, trainset.target, test_size=0.2)

bayes_multinomial = MultinomialNB().fit(train_X, train_Y)
predicted = bayes_multinomial.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[ ]:
