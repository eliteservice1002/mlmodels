#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


# In[2]:


with open("1", "r") as fopen:
    gunicorn_1 = list(filter(None, fopen.read().split("\n")))
gunicorn_1[:5]


# In[3]:


with open("2", "r") as fopen:
    gunicorn_2 = list(filter(None, fopen.read().split("\n")))
gunicorn_2[:5]


# In[4]:


with open("5", "r") as fopen:
    gunicorn_5 = list(filter(None, fopen.read().split("\n")))
gunicorn_5[:5]


# In[5]:


with open("7", "r") as fopen:
    gunicorn_7 = list(filter(None, fopen.read().split("\n")))
gunicorn_7[:5]


# In[6]:


with open("10", "r") as fopen:
    gunicorn_10 = list(filter(None, fopen.read().split("\n")))
gunicorn_10[:5]


# In[7]:


def convert_to_tuple(gunicorn):
    for i in range(len(gunicorn)):
        thread = int(gunicorn[i].split(", ")[0].split()[1])
        sec = float(gunicorn[i].split(", ")[1].split()[-2])
        gunicorn[i] = (thread, sec)
    return gunicorn


# In[8]:


gunicorn_1 = convert_to_tuple(gunicorn_1)
gunicorn_2 = convert_to_tuple(gunicorn_2)
gunicorn_5 = convert_to_tuple(gunicorn_5)
gunicorn_7 = convert_to_tuple(gunicorn_7)
gunicorn_10 = convert_to_tuple(gunicorn_10)


# In[9]:


gunicorn_1 = sorted(gunicorn_1, key=lambda x: x[0])
gunicorn_2 = sorted(gunicorn_2, key=lambda x: x[0])
gunicorn_5 = sorted(gunicorn_5, key=lambda x: x[0])
gunicorn_7 = sorted(gunicorn_7, key=lambda x: x[0])
gunicorn_10 = sorted(gunicorn_10, key=lambda x: x[0])


# In[10]:


gunicorn_1 = np.array(gunicorn_1)
gunicorn_2 = np.array(gunicorn_2)
gunicorn_5 = np.array(gunicorn_5)
gunicorn_7 = np.array(gunicorn_7)
gunicorn_10 = np.array(gunicorn_10)


# In[11]:


plt.figure(figsize=(20, 10))
plt.plot(gunicorn_1[:, 0], gunicorn_1[:, 1], label="worker: 1")
plt.plot(gunicorn_2[:, 0], gunicorn_2[:, 1], label="worker: 2")
plt.plot(gunicorn_5[:, 0], gunicorn_5[:, 1], label="worker: 5")
plt.plot(gunicorn_7[:, 0], gunicorn_7[:, 1], label="worker: 7")
plt.plot(gunicorn_10[:, 0], gunicorn_10[:, 1], label="worker: 10")
plt.legend()
plt.ylabel("time (s)")
plt.xlabel("thread")
plt.show()
