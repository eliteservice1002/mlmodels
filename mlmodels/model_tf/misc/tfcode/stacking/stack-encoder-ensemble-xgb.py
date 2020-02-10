#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import *
from sklearn.preprocessing import MinMaxScaler

import model
import xgboost as xgb

sns.set()


# In[2]:


df = pd.read_csv("GOOG-year.csv")
date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
df.head()


# In[3]:


minmax = MinMaxScaler().fit(df.iloc[:, 3].values.reshape((-1, 1)))
close_normalize = minmax.transform(df.iloc[:, 3].values.reshape((-1, 1))).reshape((-1))


# In[4]:


close_normalize.shape


# In[5]:


class encoder:
    def __init__(self, input_, dimension=2, learning_rate=0.01, hidden_layer=256, epoch=20):
        input_size = input_.shape[1]
        self.X = tf.placeholder("float", [None, input_.shape[1]])

        weights = {
            "encoder_h1": tf.Variable(tf.random_normal([input_size, hidden_layer])),
            "encoder_h2": tf.Variable(tf.random_normal([hidden_layer, dimension])),
            "decoder_h1": tf.Variable(tf.random_normal([dimension, hidden_layer])),
            "decoder_h2": tf.Variable(tf.random_normal([hidden_layer, input_size])),
        }

        biases = {
            "encoder_b1": tf.Variable(tf.random_normal([hidden_layer])),
            "encoder_b2": tf.Variable(tf.random_normal([dimension])),
            "decoder_b1": tf.Variable(tf.random_normal([hidden_layer])),
            "decoder_b2": tf.Variable(tf.random_normal([input_size])),
        }

        first_layer_encoder = tf.nn.sigmoid(
            tf.add(tf.matmul(self.X, weights["encoder_h1"]), biases["encoder_b1"])
        )
        self.second_layer_encoder = tf.nn.sigmoid(
            tf.add(tf.matmul(first_layer_encoder, weights["encoder_h2"]), biases["encoder_b2"])
        )
        first_layer_decoder = tf.nn.sigmoid(
            tf.add(
                tf.matmul(self.second_layer_encoder, weights["decoder_h1"]), biases["decoder_b1"]
            )
        )
        second_layer_decoder = tf.nn.sigmoid(
            tf.add(tf.matmul(first_layer_decoder, weights["decoder_h2"]), biases["decoder_b2"])
        )
        self.cost = tf.reduce_mean(tf.pow(self.X - second_layer_decoder, 2))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        for i in range(epoch):
            last_time = time.time()
            _, loss = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: input_})
            if (i + 1) % 10 == 0:
                print("epoch:", i + 1, "loss:", loss, "time:", time.time() - last_time)

    def encode(self, input_):
        return self.sess.run(self.second_layer_encoder, feed_dict={self.X: input_})


# In[6]:


tf.reset_default_graph()
Encoder = encoder(close_normalize.reshape((-1, 1)), 32, 0.01, 128, 100)
thought_vector = Encoder.encode(close_normalize.reshape((-1, 1)))
thought_vector.shape


# In[7]:


ada = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)
bagging = BaggingRegressor(n_estimators=500)
et = ExtraTreesRegressor(n_estimators=500)
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1)
rf = RandomForestRegressor(n_estimators=500)


# In[8]:


ada.fit(thought_vector[:-1, :], close_normalize[1:])
bagging.fit(thought_vector[:-1, :], close_normalize[1:])
et.fit(thought_vector[:-1, :], close_normalize[1:])
gb.fit(thought_vector[:-1, :], close_normalize[1:])
rf.fit(thought_vector[:-1, :], close_normalize[1:])


# In[9]:


plt.bar(np.arange(32), ada.feature_importances_)
plt.title("ada boost important feature")
plt.show()


# In[10]:


plt.bar(np.arange(32), et.feature_importances_)
plt.title("et important feature")
plt.show()


# In[11]:


plt.bar(np.arange(32), gb.feature_importances_)
plt.title("gb important feature")
plt.show()


# In[12]:


plt.bar(np.arange(32), rf.feature_importances_)
plt.title("rf important feature")
plt.show()


# In[13]:


ada_pred = ada.predict(thought_vector)
bagging_pred = bagging.predict(thought_vector)
et_pred = et.predict(thought_vector)
gb_pred = gb.predict(thought_vector)
rf_pred = rf.predict(thought_vector)


# In[14]:


ada_actual = np.hstack([close_normalize[0], ada_pred[:-1]])
bagging_actual = np.hstack([close_normalize[0], bagging_pred[:-1]])
et_actual = np.hstack([close_normalize[0], et_pred[:-1]])
gb_actual = np.hstack([close_normalize[0], gb_pred[:-1]])
rf_actual = np.hstack([close_normalize[0], rf_pred[:-1]])
stack_predict = np.vstack(
    [ada_actual, bagging_actual, et_actual, gb_actual, rf_actual, close_normalize]
).T
corr_df = pd.DataFrame(stack_predict)


# In[15]:


sns.heatmap(corr_df.corr(), annot=True)
plt.show()


# # Wow, I do not expect this heatmap. Totally a heat!

# In[16]:


params_xgd = {
    "max_depth": 7,
    "objective": "reg:logistic",
    "learning_rate": 0.05,
    "n_estimators": 10000,
}
train_Y = close_normalize[1:]
clf = xgb.XGBRegressor(**params_xgd)
clf.fit(
    stack_predict[:-1, :],
    train_Y,
    eval_set=[(stack_predict[:-1, :], train_Y)],
    eval_metric="rmse",
    early_stopping_rounds=20,
    verbose=False,
)


# In[17]:


xgb_pred = clf.predict(stack_predict)
xgb_actual = np.hstack([close_normalize[0], xgb_pred[:-1]])
date_original = pd.Series(date_ori).dt.strftime(date_format="%Y-%m-%d").tolist()


# In[18]:


def reverse_close(array):
    return minmax.inverse_transform(array.reshape((-1, 1))).reshape((-1))


# In[25]:


plt.figure(figsize=(15, 6))
x_range = np.arange(df.Close.shape[0])
plt.plot(x_range, df.Close, label="Real Close")
plt.plot(x_range, reverse_close(ada_pred), label="ada Close")
plt.plot(x_range, reverse_close(bagging_pred), label="bagging Close")
plt.plot(x_range, reverse_close(et_pred), label="et Close")
plt.plot(x_range, reverse_close(gb_pred), label="gb Close")
plt.plot(x_range, reverse_close(rf_pred), label="rf Close")
plt.plot(x_range, reverse_close(xgb_pred), label="xgb stacked Close")
plt.legend()
plt.xticks(x_range[::50], date_original[::50])
plt.title("stacked")
plt.show()


# In[20]:


ada_list = ada_pred.tolist()
bagging_list = bagging_pred.tolist()
et_list = et_pred.tolist()
gb_list = gb_pred.tolist()
rf_list = rf_pred.tolist()
xgb_list = xgb_pred.tolist()


def predict(count, history=5):
    for i in range(count):
        roll = np.array(xgb_list[-history:])
        thought_vector = Encoder.encode(roll.reshape((-1, 1)))
        ada_pred = ada.predict(thought_vector)
        bagging_pred = bagging.predict(thought_vector)
        et_pred = et.predict(thought_vector)
        gb_pred = gb.predict(thought_vector)
        rf_pred = rf.predict(thought_vector)
        ada_list.append(ada_pred[-1])
        bagging_list.append(bagging_pred[-1])
        et_list.append(et_pred[-1])
        gb_list.append(gb_pred[-1])
        rf_list.append(rf_pred[-1])
        ada_actual = np.hstack([xgb_list[-history], ada_pred[:-1]])
        bagging_actual = np.hstack([xgb_list[-history], bagging_pred[:-1]])
        et_actual = np.hstack([xgb_list[-history], et_pred[:-1]])
        gb_actual = np.hstack([xgb_list[-history], gb_pred[:-1]])
        rf_actual = np.hstack([xgb_list[-history], rf_pred[:-1]])
        stack_predict = np.vstack(
            [ada_actual, bagging_actual, et_actual, gb_actual, rf_actual, xgb_list[-history:]]
        ).T
        xgb_pred = clf.predict(stack_predict)
        xgb_list.append(xgb_pred[-1])
        date_ori.append(date_ori[-1] + timedelta(days=1))


# In[21]:


predict(30, history=5)


# In[29]:


plt.figure(figsize=(15, 6))
x_range = np.arange(df.Close.shape[0])
x_range_future = np.arange(len(xgb_list))
plt.plot(x_range, df.Close, label="Real Close")
plt.plot(x_range_future, reverse_close(np.array(ada_list)), label="ada Close")
plt.plot(x_range_future, reverse_close(np.array(bagging_list)), label="bagging Close")
plt.plot(x_range_future, reverse_close(np.array(et_list)), label="et Close")
plt.plot(x_range_future, reverse_close(np.array(gb_list)), label="gb Close")
plt.plot(x_range_future, reverse_close(np.array(rf_list)), label="rf Close")
plt.plot(x_range_future, reverse_close(np.array(xgb_list)), label="xgb stacked Close")
plt.legend()
plt.xticks(
    x_range_future[::50], pd.Series(date_ori).dt.strftime(date_format="%Y-%m-%d").tolist()[::50]
)
plt.title("stacked")
plt.show()


# In[ ]:
