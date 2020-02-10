#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import pickle
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.datasets
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.util import skipgrams
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

# In[2]:


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


# In[3]:


trainset = sklearn.datasets.load_files(container_path="data", encoding="UTF-8")
trainset.data, trainset.target = separate_dataset(trainset)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[4]:


cv = StratifiedKFold(n_splits=10, shuffle=True)
stemmer = PorterStemmer()

stopwords = stopwords.words("english")
other_exclusions = ["ff", "rt"]
stopwords.extend(other_exclusions)


def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    # return [token.strip() for token in tweet.split()]
    # tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    # tweet = " ".join(re.split("[^^a-zA-Z.,!?]*", tweet)).strip()
    tweet = " ".join(re.split("[^a-zA-Z#]+", tweet)).strip()
    # tweet = " ".join(re.split("[ ]*", tweet)).strip()
    return tweet.split()


def tokenize(tweet):
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens


def get_metric(vectorizer, X_raw, y_raw, name):
    result = {"name": name}
    y = y_raw
    X = vectorizer.fit_transform(X_raw)
    result["shape"] = X.shape

    aucs = []
    for train, test in cv.split(X, y):
        classifier.fit(X[train], y[train])
        y_preds = classifier.predict(X[test])
        accuracy = accuracy_score(y[test], y_preds)
        aucs.append(accuracy)

    result["accuracies"] = aucs
    result["mean_accuracy"] = np.mean(aucs)
    # result['y_preds'] = y_preds
    return result


# In[9]:


classifier = LinearSVC(C=1)
vectorizer_unigrams = TfidfVectorizer(
    ngram_range=(1, 1), stop_words=other_exclusions, tokenizer=basic_tokenize
)

result = get_metric(
    vectorizer_unigrams, np.array(trainset.data), np.array(trainset.target), "unigrams-basic"
)
result


# In[10]:


vectorizer_bigrams = TfidfVectorizer(
    ngram_range=(2, 2), stop_words=other_exclusions, tokenizer=basic_tokenize
)

result = get_metric(
    vectorizer_bigrams, np.array(trainset.data), np.array(trainset.target), "bigrams-basic"
)
result


# In[11]:


vectorizer_trigrams = TfidfVectorizer(
    ngram_range=(3, 3), stop_words=other_exclusions, tokenizer=basic_tokenize
)

result = get_metric(
    vectorizer_trigrams, np.array(trainset.data), np.array(trainset.target), "trigrams-basic"
)
result


# In[13]:


def skipgram_tokenize(tweet, n=None, k=None, include_all=True):
    tokens = [w for w in basic_tokenize(tweet)]
    if include_all:
        result = []
        for i in range(k + 1):
            skg = [w for w in skipgrams(tokens, n, i)]
            result = result + skg
    else:
        result = [w for w in skipgrams(tokens, n, k)]
    return result


def make_skip_tokenize(n, k, include_all=True):
    return lambda tweet: skipgram_tokenize(tweet, n=n, k=k, include_all=include_all)


# In[14]:


vectorizer_1skipbigram = TfidfVectorizer(
    stop_words=other_exclusions, tokenizer=make_skip_tokenize(n=2, k=1)
)

result = get_metric(
    vectorizer_1skipbigram,
    np.array(trainset.data),
    np.array(trainset.target),
    "1-skip-bigrams-basic",
)
result


# In[15]:


vectorizer_2skipbigram = TfidfVectorizer(
    stop_words=other_exclusions, tokenizer=make_skip_tokenize(n=2, k=2)
)

result = get_metric(
    vectorizer_2skipbigram,
    np.array(trainset.data),
    np.array(trainset.target),
    "2-skip-bigrams-basic",
)
result


# In[16]:


vectorizer_3skipbigram = TfidfVectorizer(
    stop_words=other_exclusions, tokenizer=make_skip_tokenize(n=2, k=3)
)
result = get_metric(
    vectorizer_3skipbigram,
    np.array(trainset.data),
    np.array(trainset.target),
    "3-skip-bigrams-basic",
)
result


# In[17]:


vectorizer_character_bigram = TfidfVectorizer(
    stop_words=other_exclusions, analyzer="char", ngram_range=(2, 2)
)
result = get_metric(
    vectorizer_character_bigram,
    np.array(trainset.data),
    np.array(trainset.target),
    "character bigrams",
)
result


# In[18]:


vectorizer_character_trigram = TfidfVectorizer(
    stop_words=other_exclusions, analyzer="char", ngram_range=(3, 3)
)
result = get_metric(
    vectorizer_character_trigram,
    np.array(trainset.data),
    np.array(trainset.target),
    "character trigrams",
)
result


# In[19]:


vectorizer_character_4gram = TfidfVectorizer(
    stop_words=other_exclusions, analyzer="char", ngram_range=(4, 4)
)
result = get_metric(
    vectorizer_character_4gram,
    np.array(trainset.data),
    np.array(trainset.target),
    "character 4-grams",
)
result


# In[20]:


vectorizer_character_5gram = TfidfVectorizer(
    stop_words=other_exclusions, analyzer="char", ngram_range=(5, 5)
)
result = get_metric(
    vectorizer_character_5gram,
    np.array(trainset.data),
    np.array(trainset.target),
    "character 5-grams",
)
result


# In[21]:


vectorizer_character_6gram = TfidfVectorizer(
    stop_words=other_exclusions, analyzer="char", ngram_range=(6, 6)
)
result = get_metric(
    vectorizer_character_6gram,
    np.array(trainset.data),
    np.array(trainset.target),
    "character 6-grams",
)
result


# In[22]:


vectorizer_character_7gram = TfidfVectorizer(
    stop_words=other_exclusions, analyzer="char", ngram_range=(7, 7)
)

result = get_metric(
    vectorizer_character_7gram,
    np.array(trainset.data),
    np.array(trainset.target),
    "character 7-grams",
)
result


# In[23]:


vectorizer_character_8gram = TfidfVectorizer(
    stop_words=other_exclusions, analyzer="char", ngram_range=(8, 8)
)
result = get_metric(
    vectorizer_character_8gram,
    np.array(trainset.data),
    np.array(trainset.target),
    "character 8-grams",
)
result


# In[24]:


def get_metric_oracle(X_raw, y_raw, vectorizers):
    results = {"oracle": {}}
    for train, test in cv.split(X_raw, y_raw):
        y_train = y_raw[train]
        X_train = X_raw[train]

        y_test = y_raw[test]
        X_test = X_raw[test]

        y_pred_oracle = []
        for name in vectorizers:
            vectorizer = vectorizers[name]
            if name in results:
                result = results[name]
            else:
                result = {}
                results[name] = result

            X_train_tr = vectorizer.fit_transform(X_train)

            if not "shape" in result:
                result["shape"] = []
            result["shape"].append(X_train_tr.shape)
            classifier.fit(X_train_tr, y_train)
            X_test_tr = vectorizer.transform(X_test)
            y_preds = classifier.predict(X_test_tr)
            accuracy = accuracy_score(y_test, y_preds)

            if not "accuracies" in result:
                result["accuracies"] = []

            result["accuracies"].append(accuracy)

            if not "y_preds" in result:
                result["y_preds"] = []

            result["y_preds"].append(y_preds)

            y_pred_oracle.append(y_preds)

        y_pred_oracle = np.matrix(y_pred_oracle).T
        oracle_correct_pred = 0
        oracle_incorrect_index = []
        for i, yt in enumerate(y_test):
            if True in (y_pred_oracle[i, :] == yt):
                oracle_correct_pred += 1
            else:
                oracle_incorrect_index.append(test[i])

        accuracy = oracle_correct_pred / len(y_test)
        print("Oracle classifier accuracy={}".format(accuracy))
        result = results["oracle"]

        if not "accuracies" in result:
            result["accuracies"] = []

        result["accuracies"].append(accuracy)

        if not "oracle_incorrect_index" in result:
            result["oracle_incorrect_index"] = []

        result["oracle_incorrect_index"] = oracle_incorrect_index
    return results


# In[25]:


vectorizers = {
    "vectorizer_character_8gram": vectorizer_character_8gram,
    "vectorizer_character_7gram": vectorizer_character_7gram,
    "vectorizer_character_6gram": vectorizer_character_6gram,
    "vectorizer_character_5gram": vectorizer_character_5gram,
    "vectorizer_character_4gram": vectorizer_character_4gram,
    "vectorizer_1skipbigram": vectorizer_1skipbigram,
    "vectorizer_2skipbigram": vectorizer_2skipbigram,
    "vectorizer_3skipbigram": vectorizer_3skipbigram,
    "vectorizer_unigrams": vectorizer_unigrams,
    "vectorizer_bigrams": vectorizer_bigrams,
    "vectorizer_trigrams": vectorizer_trigrams,
}

results = get_metric_oracle(np.array(trainset.data), np.array(trainset.target), vectorizers)


# In[26]:


incorrect_indexes = sorted(set(results["oracle"]["oracle_incorrect_index"]))
print(len(incorrect_indexes))


# In[28]:


X_incorrect = np.array(trainset.data)[incorrect_indexes]
y_incorrect = np.array(trainset.target)[incorrect_indexes]
incorrect_classified = pd.DataFrame()
incorrect_classified["text"] = X_incorrect
incorrect_classified["label"] = y_incorrect
incorrect_classified


# In[29]:


incorrect_classified.label.value_counts()


# In[30]:


summary = []
for name in results:
    result = results[name]
    accuracies = result["accuracies"]
    summary.append({"name": name, "accuracy": np.mean(accuracies)})
df_summary = pd.DataFrame(summary)
df_summary = df_summary.sort_values(by=["accuracy"], ascending=False)
df_summary = df_summary.reset_index()
df_summary


# In[ ]:
