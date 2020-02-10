#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import pickle
import re

import numpy as np

# In[7]:


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

    return dataset


# In[8]:


dataset = read_data("data/")
dataset[:5, :]


# In[ ]:


with open("dataset-emotion.p", "wb") as fopen:
    pickle.dump(dataset, fopen)
