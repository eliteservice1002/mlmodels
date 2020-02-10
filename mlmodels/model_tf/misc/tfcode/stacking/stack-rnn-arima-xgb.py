#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime, timedelta
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import tensorflow as tf
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

import autoencoder
import model
import xgboost as xgb
from xgboost import plot_importance

sns.set()


# ## Deep Feed-forward Auto-Encoder Neural Network  to reduce dimension + Deep Recurrent Neural Network + ARIMA + Extreme Boosting Gradient Regressor

# ### Our target is Close market

# In[2]:


google = pd.read_csv("GOOG.csv")
eur_myr = pd.read_csv("eur-myr.csv")
usd_myr = pd.read_csv("usd-myr.csv")
oil = pd.read_csv("oil.csv")


# In[3]:


google["oil_price"] = oil["Price"]
google["oil_open"] = oil["Open"]
google["oil_high"] = oil["High"]
google["oil_low"] = oil["Low"]
google["eur_myr"] = eur_myr["Unnamed: 1"]
google["usd_myr"] = usd_myr["Unnamed: 1"]


# In[4]:


date_ori = pd.to_datetime(google.iloc[:, 0]).tolist()
google.head()


# In[5]:


minmax = MinMaxScaler().fit(google.iloc[:, 4].values.reshape((-1, 1)))
df_log = MinMaxScaler().fit_transform(google.iloc[:, 1:].astype("float32"))
df_log = pd.DataFrame(df_log)
df_log.head()


# In[6]:


thought_vector = autoencoder.reducedimension(df_log.values, 4, 0.001, 128, 100)


# In[7]:


thought_vector.shape


# In[8]:


num_layers = 1
size_layer = 128
timestamp = 5
epoch = 500
dropout_rate = 0.1


# In[9]:


tf.reset_default_graph()
modelnn = model.Model(0.01, num_layers, thought_vector.shape[1], size_layer, 1, dropout_rate)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(epoch):
    init_value = np.zeros((1, num_layers * 2 * size_layer))
    total_loss = 0
    for k in range(0, (thought_vector.shape[0] // timestamp) * timestamp, timestamp):
        batch_x = np.expand_dims(thought_vector[k : k + timestamp, :], axis=0)
        batch_y = df_log.values[k + 1 : k + timestamp + 1, 3].reshape([-1, 1])
        last_state, _, loss = sess.run(
            [modelnn.last_state, modelnn.optimizer, modelnn.cost],
            feed_dict={modelnn.X: batch_x, modelnn.Y: batch_y, modelnn.hidden_layer: init_value},
        )
        init_value = last_state
        total_loss += loss
    total_loss /= thought_vector.shape[0] // timestamp
    if (i + 1) % 100 == 0:
        print("epoch:", i + 1, "avg loss:", total_loss)


# In[10]:


output_predict = np.zeros(((thought_vector.shape[0] // timestamp) * timestamp, 1))
init_value = np.zeros((1, num_layers * 2 * size_layer))
for k in range(0, (thought_vector.shape[0] // timestamp) * timestamp, timestamp):
    out_logits, last_state = sess.run(
        [modelnn.logits, modelnn.last_state],
        feed_dict={
            modelnn.X: np.expand_dims(thought_vector[k : k + timestamp, :], axis=0),
            modelnn.hidden_layer: init_value,
        },
    )
    init_value = last_state
    output_predict[k : k + timestamp, :] = out_logits


# In[11]:


print(
    "Mean Square Error:",
    np.mean(
        np.square(
            output_predict[:, 0]
            - df_log.iloc[1 : (thought_vector.shape[0] // timestamp) * timestamp + 1, 0].values
        )
    ),
)


# Import ARIMA model using stats model

# In[12]:


Qs = range(0, 1)
qs = range(0, 2)
Ps = range(0, 2)
ps = range(0, 2)
D = 1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
best_aic = float("inf")
for param in parameters_list:
    try:
        arima = sm.tsa.statespace.SARIMAX(
            df_log.iloc[:, 3].values,
            order=(param[0], D, param[1]),
            seasonal_order=(param[2], D, param[3], 1),
        ).fit(disp=-1)
    except:
        continue
    aic = arima.aic
    if aic < best_aic and aic:
        best_arima = arima
        best_aic = aic

best_aic


# In[13]:


def reverse_close(array):
    return minmax.inverse_transform(array.reshape((-1, 1))).reshape((-1))


# In[14]:


pred_arima = best_arima.predict()
x_range = np.arange(df_log.shape[0])
fig = plt.figure(figsize=(15, 6))
ax = plt.subplot(111)
ax.plot(x_range, reverse_close(df_log.iloc[:, 3].values), label="true Close")
ax.plot(x_range, reverse_close(pred_arima), label="predict Close using Arima")
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.xticks(x_range[::5], date_ori[::5])
plt.title("overlap market Close")
plt.show()


# In[15]:


boundary = (thought_vector.shape[0] // timestamp) * timestamp
stack_predict = np.vstack([pred_arima[:boundary], output_predict.reshape((-1))]).T


# In[16]:


where_below_0 = np.where(stack_predict < 0)
where_higher_1 = np.where(stack_predict > 1)
stack_predict[where_below_0[0], where_below_0[1]] = 0
stack_predict[where_higher_1[0], where_higher_1[1]] = 1


# In[17]:


corr_df = pd.DataFrame(np.hstack([stack_predict, df_log.values[:boundary, 3].reshape((-1, 1))]))


# In[18]:


sns.heatmap(corr_df.corr(), annot=True)
plt.show()


# ARIMA able to predict data that correlate 0.61 originally from original Close
#
# Deep Recurrent Neural Network able to predict data that correlate 0.48 originally from original Close

# In[19]:


params_xgd = {
    "max_depth": 7,
    "objective": "reg:logistic",
    "learning_rate": 0.05,
    "n_estimators": 10000,
}
train_Y = df_log.values[:boundary, 3]
clf = xgb.XGBRegressor(**params_xgd)
clf.fit(
    stack_predict,
    train_Y,
    eval_set=[(stack_predict, train_Y)],
    eval_metric="rmse",
    early_stopping_rounds=20,
    verbose=False,
)


# In[20]:


stacked = clf.predict(stack_predict)


# In[25]:


plt.figure(figsize=(15, 6))
x_range = np.arange(boundary)
plt.plot(x_range, reverse_close(train_Y), label="Real Close")
plt.plot(x_range, reverse_close(pred_arima[:boundary]), label="ARIMA Close")
plt.plot(x_range, reverse_close(output_predict), label="RNN Close")
plt.plot(x_range, reverse_close(stacked), label="Stacked Close")
plt.legend()
plt.xticks(x_range[::5], date_ori[:boundary][::5])
plt.title("stacked RNN + ARIMA with XGB")
plt.show()


# # Pretty insane i can say!

# In[26]:


plot_importance(clf)
plt.show()


# ## Arima is more important than RNN
