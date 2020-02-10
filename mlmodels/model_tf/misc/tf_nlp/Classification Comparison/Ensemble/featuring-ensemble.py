#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import re
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import sklearn.datasets
from nltk.corpus import stopwords
from sklearn import ensemble, metrics, model_selection, naive_bayes
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import spacy
import xgboost as xgb

color = sns.color_palette()

get_ipython().run_line_magic("matplotlib", "inline")

eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None
nlp = spacy.load("en_core_web_sm")


# In[2]:


def clearstring(string):
    string = re.sub("[^A-Za-z ]+", "", string)
    string = string.split(" ")
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = " ".join(string)
    return string.lower()


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


# In[3]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset)
combined = list(zip(trainset.data, trainset.target))
random.shuffle(combined)
trainset.data[:], trainset.target[:] = zip(*combined)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[4]:


df = pd.DataFrame({"text": trainset.data, "label": trainset.target})

df["num_words"] = df["text"].apply(lambda x: len(str(x).split()))

df["num_unique_words"] = df["text"].apply(lambda x: len(set(str(x).split())))

df["num_chars"] = df["text"].apply(lambda x: len(str(x)))

df["num_stopwords"] = df["text"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)

df["mean_word_len"] = df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


# In[5]:


df_train = df.iloc[: int(df.shape[0] * 0.8), :]
df_test = df.iloc[int(df.shape[0] * 0.8) :, :]
df_test = df_test.reset_index()
df_test = df_test.iloc[:, 1:]
print(df_train.shape)
print(df_test.shape)


# In[6]:


param = {}
param["objective"] = "multi:softprob"
param["learning_rate"] = 0.1
param["max_depth"] = 3
param["silent"] = 1
param["subsample"] = 0.8
feature_clf = xgb.XGBClassifier(**param)
train_X = df_train.iloc[:, 2:]
train_Y = df_train.iloc[:, 0]
test_X = df_test.iloc[:, 2:]
test_Y = df_test.iloc[:, 0]
feature_clf.fit(
    train_X,
    train_Y,
    eval_set=[(test_X, test_Y)],
    eval_metric="mlogloss",
    early_stopping_rounds=20,
    verbose=True,
)


# In[7]:


predicted = feature_clf.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[8]:


fig, ax = plt.subplots(figsize=(6, 6))
xgb.plot_importance(feature_clf, height=0.8, ax=ax)
plt.show()


# In[9]:


tfidf_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
tfidf = tfidf_vec.fit_transform(df.text)
tfidf_train = tfidf_vec.transform(df_train.text)
tfidf_test = tfidf_vec.transform(df_test.text)


# In[10]:


model_nb = naive_bayes.MultinomialNB()
model_nb.fit(tfidf_train, train_Y)
predicted = model_nb.predict(tfidf_test)
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[13]:


train_tfidf_nb = pd.DataFrame(model_nb.predict_proba(tfidf_train))
test_tfidf_nb = pd.DataFrame(model_nb.predict_proba(tfidf_test))
train_tfidf_nb.columns = ["nb_tfidf_" + i for i in trainset.target_names]
test_tfidf_nb.columns = ["nb_tfidf_" + i for i in trainset.target_names]
df_train = pd.concat([df_train, train_tfidf_nb], axis=1)
df_test = pd.concat([df_test, test_tfidf_nb], axis=1)


# In[15]:


n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm="arpack")
svd_obj.fit(tfidf)
train_svd = pd.DataFrame(svd_obj.transform(tfidf_train))
test_svd = pd.DataFrame(svd_obj.transform(tfidf_test))

train_svd.columns = ["lsa_" + str(i) for i in range(n_comp)]
test_svd.columns = ["lsa_" + str(i) for i in range(n_comp)]
df_train = pd.concat([df_train, train_svd], axis=1)
df_test = pd.concat([df_test, test_svd], axis=1)


# In[16]:


del tfidf, tfidf_train, tfidf_test, train_svd, test_svd
bow_vec = CountVectorizer(stop_words="english", ngram_range=(1, 3))
bow = bow_vec.fit_transform(df.text)
bow_train = bow_vec.transform(df_train.text)
bow_test = bow_vec.transform(df_test.text)


# In[17]:


model_nb = naive_bayes.MultinomialNB()
model_nb.fit(bow_train, train_Y)
predicted = model_nb.predict(bow_test)
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[18]:


train_bow_nb = pd.DataFrame(model_nb.predict_proba(bow_train))
test_bow_nb = pd.DataFrame(model_nb.predict_proba(bow_test))
train_bow_nb.columns = ["nb_bow_" + i for i in trainset.target_names]
test_bow_nb.columns = ["nb_bow_" + i for i in trainset.target_names]
df_train = pd.concat([df_train, train_bow_nb], axis=1)
df_test = pd.concat([df_test, test_bow_nb], axis=1)


# In[19]:


bow_vec = CountVectorizer(ngram_range=(1, 7), analyzer="char")
bow = bow_vec.fit_transform(df.text)
bow_train = bow_vec.transform(df_train.text)
bow_test = bow_vec.transform(df_test.text)


# In[21]:


model_nb = naive_bayes.MultinomialNB()
model_nb.fit(bow_train, train_Y)
predicted = model_nb.predict(bow_test)
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[22]:


train_bow_nb = pd.DataFrame(model_nb.predict_proba(bow_train))
test_bow_nb = pd.DataFrame(model_nb.predict_proba(bow_test))
train_bow_nb.columns = ["nb_bow_char_" + i for i in trainset.target_names]
test_bow_nb.columns = ["nb_bow_char_" + i for i in trainset.target_names]
df_train = pd.concat([df_train, train_bow_nb], axis=1)
df_test = pd.concat([df_test, test_bow_nb], axis=1)


# In[23]:


del bow_vec, train_bow_nb, test_bow_nb, bow, bow_train, bow_test


# In[25]:


df_train.head()


# In[30]:


params_xgd = {
    "min_child_weight": 10.0,
    "max_depth": 7,
    "objective": "multi:softprob",
    "max_delta_step": 1.8,
    "colsample_bytree": 0.4,
    "subsample": 0.8,
    "learning_rate": 0.05,
    "gamma": 0.65,
    "nthread": -1,
    "silent": False,
    "n_estimators": 10000,
}
train_X = df_train.iloc[:, 2:]
test_X = df_test.iloc[:, 2:]
train_Y = df_train.iloc[:, 0]
test_Y = df_test.iloc[:, 0]
clf = xgb.XGBClassifier(**params_xgd)
clf.fit(
    train_X,
    train_Y,
    eval_set=[(test_X, test_Y)],
    eval_metric="mlogloss",
    early_stopping_rounds=50,
    verbose=False,
)


# In[31]:


predicted = clf.predict(test_X)
print("accuracy validation set: ", np.mean(predicted == test_Y))
print(metrics.classification_report(test_Y, predicted, target_names=trainset.target_names))


# In[32]:


fig, ax = plt.subplots(figsize=(12, 12))
xgb.plot_importance(clf, height=0.8, ax=ax)
plt.show()


# In[39]:


process = psutil.Process(os.getpid())
"this notebook use: " + str(process.memory_info()[0] / float(2 ** 20)) + " MB"
