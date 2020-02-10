#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()


# In[2]:


df = pd.read_csv("TSLA.csv")
df.head()


# In[3]:


returns = df.Close.pct_change()
volatility = returns.std()


# In[4]:


number_simulation = 100
predict_day = 30
results = pd.DataFrame()

for i in range(number_simulation):
    prices = []
    prices.append(df.Close.iloc[-1])
    for d in range(predict_day):
        prices.append(prices[d] * (1 + np.random.normal(0, volatility)))
    results[i] = pd.Series(prices).values


# In[5]:


plt.figure(figsize=(10, 5))
plt.plot(results)
plt.ylabel("Value")
plt.xlabel("Simulated days")
plt.show()


# In[6]:


raveled = results.values.ravel()
raveled.sort()
cp_raveled = raveled.copy()

plt.figure(figsize=(17, 5))
plt.subplot(1, 3, 1)
sns.distplot(raveled, norm_hist=True)
plt.xlabel("Close")
plt.ylabel("probability")
plt.title("$\mu$ = %.2f, $\sigma$ = %.2f" % (raveled.mean(), raveled.std()))
plt.subplot(1, 3, 2)
sns.distplot(df.Close, norm_hist=True)
plt.title("$\mu$ = %.2f, $\sigma$ = %.2f" % (df.Close.mean(), df.Close.std()))
plt.subplot(1, 3, 3)
sns.distplot(raveled, norm_hist=True, label="monte carlo samples")
sns.distplot(df.Close, norm_hist=True, label="real samples")
plt.legend()
plt.show()


# In[7]:


returns_volume = df.Volume.pct_change()
std = returns_volume.std()
variance = std ** 2

gaussian_2d = (1 / (2 * np.pi * variance)) * np.exp(
    -1 * ((returns_volume ** 2 + returns ** 2) / (2 * variance))
).std()

print(volatility, gaussian_2d)


# In[8]:


for i in range(number_simulation):
    prices = []
    prices.append(df.Close.iloc[-1])
    for d in range(predict_day):
        prices.append(prices[d] * (1 + np.random.normal(0, gaussian_2d)))
    results[i] = pd.Series(prices).values


# In[9]:


plt.figure(figsize=(10, 5))
plt.plot(results)
plt.ylabel("Value")
plt.xlabel("Simulated days")
plt.show()


# In[10]:


raveled = results.values.ravel()
raveled.sort()

plt.figure(figsize=(17, 5))
plt.subplot(1, 3, 1)
sns.distplot(raveled, norm_hist=True)
plt.xlabel("Close")
plt.ylabel("probability")
plt.title("$\mu$ = %.2f, $\sigma$ = %.2f" % (raveled.mean(), raveled.std()))
plt.subplot(1, 3, 2)
sns.distplot(df.Close, norm_hist=True)
plt.title("$\mu$ = %.2f, $\sigma$ = %.2f" % (df.Close.mean(), df.Close.std()))
plt.subplot(1, 3, 3)
sns.distplot(raveled, norm_hist=True, label="monte carlo samples")
sns.distplot(df.Close, norm_hist=True, label="real samples")
plt.legend()
plt.show()


# In[11]:


def pct_change(x, period=1):
    x = np.array(x)
    return (x[period:] - x[:-period]) / x[:-period]


# In[12]:


results = pd.DataFrame()

for i in range(number_simulation):
    prices = df.Close.values[-predict_day:].tolist()
    volatility = pct_change(prices[-predict_day:]).std()
    for d in range(predict_day):
        prices.append(prices[-1] * (1 + np.random.normal(0, volatility)))
        volatility = pct_change(prices[-predict_day:]).std()
    results[i] = pd.Series(prices[-predict_day:]).values


# In[13]:


plt.figure(figsize=(10, 5))
plt.plot(results)
plt.ylabel("Value")
plt.xlabel("Simulated days")
plt.show()


# In[14]:


raveled = results.values.ravel()
raveled.sort()

plt.figure(figsize=(17, 5))
plt.subplot(1, 3, 1)
sns.distplot(raveled, norm_hist=True, label="monte carlo samples")
sns.distplot(cp_raveled, norm_hist=True, label="constant volatility samples")
plt.legend()
plt.xlabel("Close")
plt.ylabel("probability")
plt.title("$\mu$ = %.2f, $\sigma$ = %.2f" % (raveled.mean(), raveled.std()))
plt.subplot(1, 3, 2)
sns.distplot(df.Close, norm_hist=True)
plt.title("$\mu$ = %.2f, $\sigma$ = %.2f" % (df.Close.mean(), df.Close.std()))
plt.subplot(1, 3, 3)
sns.distplot(raveled, norm_hist=True, label="monte carlo samples")
sns.distplot(df.Close, norm_hist=True, label="real samples")
sns.distplot(cp_raveled, norm_hist=True, label="constant volatility samples")
plt.legend()
plt.show()


# In[ ]:
