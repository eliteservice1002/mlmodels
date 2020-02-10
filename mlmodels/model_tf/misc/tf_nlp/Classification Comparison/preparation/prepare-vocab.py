#!/usr/bin/env python
# coding: utf-8

# In[7]:


import collections
import os
import pickle
import re

import numpy as np

# In[11]:


def clearstring(string):
    string = re.sub("[^'\"A-Za-z0-9 ]+", "", string)
    string = string.split(" ")
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = [y for y in string if len(y) > 3 and y.find("nbsp") < 0]
    return " ".join(string)


def read_data(location):
    list_folder = os.listdir(location)
    label = list_folder
    label.sort()
    outer_string, outer_label = [], []
    for i in range(len(list_folder)):
        list_file = os.listdir("data/" + list_folder[i])
        strings = []
        for x in range(len(list_file)):
            with open("data/" + list_folder[i] + "/" + list_file[x], "r") as fopen:
                strings += fopen.read().split("\n")
        strings = list(filter(None, strings))
        for k in range(len(strings)):
            strings[k] = clearstring(strings[k])
        labels = [i] * len(strings)
        outer_string += strings
        outer_label += labels

    dataset = np.array([outer_string, outer_label])
    dataset = dataset.T
    np.random.shuffle(dataset)

    string = []
    for i in range(dataset.shape[0]):
        string += dataset[i][0].split()

    return string


def build_vocab(words, n_words):
    count = [["UNK", -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# In[9]:


strings = read_data("data")


# In[10]:


strings[:5]


# In[12]:


n_words = len(set(strings))
_, _, dictionary, reversed_dictionary = build_vocab(strings, n_words)


# In[ ]:


with open("dataset-dictionary.p", "wb") as fopen:
    pickle.dump(reversed_dictionary, fopen)
with open("dataset-dictionary-reverse.p", "wb") as fopen:
    pickle.dump(dictionary, fopen)
