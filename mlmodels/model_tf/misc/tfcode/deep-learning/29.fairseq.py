#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

sns.set()


# In[2]:


df = pd.read_csv("../dataset/GOOG-year.csv")
date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
df.head()


# In[3]:


minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype("float32"))
df_log = minmax.transform(df.iloc[:, 1:].astype("float32"))
df_log = pd.DataFrame(df_log)
df_log.head()


# In[4]:


emb_size = 128
n_hidden = 128
n_layers = 1
n_attn_heads = 16
learning_rate = 1e-3
timestamp = 5
epoch = 500
future_day = 50
size = df_log.shape[1]
output_size = df_log.shape[1]


# In[5]:


def encoder_block(inp, n_hidden, filter_size):
    inp = tf.expand_dims(inp, 2)
    inp = tf.pad(
        inp, [[0, 0], [(filter_size[0] - 1) // 2, (filter_size[0] - 1) // 2], [0, 0], [0, 0]]
    )
    conv = tf.layers.conv2d(inp, n_hidden, filter_size, padding="VALID", activation=None)
    conv = tf.squeeze(conv, 2)
    return conv


def decoder_block(inp, n_hidden, filter_size):
    inp = tf.expand_dims(inp, 2)
    inp = tf.pad(inp, [[0, 0], [filter_size[0] - 1, 0], [0, 0], [0, 0]])
    conv = tf.layers.conv2d(inp, n_hidden, filter_size, padding="VALID", activation=None)
    conv = tf.squeeze(conv, 2)
    return conv


def glu(x):
    return tf.multiply(x[:, :, : tf.shape(x)[2] // 2], tf.sigmoid(x[:, :, tf.shape(x)[2] // 2 :]))


def layer(inp, conv_block, kernel_width, n_hidden, residual=None):
    z = conv_block(inp, n_hidden, (kernel_width, 1))
    return glu(z) + (residual if residual is not None else 0)


class Fairseq:
    def __init__(self):
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))

        encoder_embedded = tf.layers.dense(self.X, emb_size)
        encoder_embedded = tf.nn.dropout(encoder_embedded, keep_prob=0.75)

        e = tf.identity(encoder_embedded)
        for i in range(n_layers):
            z = layer(encoder_embedded, encoder_block, 3, n_hidden * 2, encoder_embedded)
            encoder_embedded = z

        encoder_output, output_memory = z, z + e
        g = tf.identity(encoder_embedded)

        for i in range(n_layers):
            attn_res = h = layer(
                encoder_embedded,
                decoder_block,
                3,
                n_hidden * 2,
                residual=tf.zeros_like(encoder_embedded),
            )
            C = []
            for j in range(n_attn_heads):
                h_ = tf.layers.dense(h, n_hidden // n_attn_heads)
                g_ = tf.layers.dense(g, n_hidden // n_attn_heads)
                zu_ = tf.layers.dense(encoder_output, n_hidden // n_attn_heads)
                ze_ = tf.layers.dense(output_memory, n_hidden // n_attn_heads)

                d = tf.layers.dense(h_, n_hidden // n_attn_heads) + g_
                dz = tf.matmul(d, tf.transpose(zu_, [0, 2, 1]))
                a = tf.nn.softmax(dz)
                c_ = tf.matmul(a, ze_)
                C.append(c_)

            c = tf.concat(C, 2)
            h = tf.layers.dense(attn_res + c, n_hidden)
            encoder_embedded = h

        encoder_embedded = tf.sigmoid(h)
        self.logits = tf.layers.dense(encoder_embedded[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


# In[6]:


tf.reset_default_graph()
modelnn = Fairseq()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


# In[7]:


for i in range(epoch):
    total_loss = 0
    for k in range(0, df_log.shape[0] - 1, timestamp):
        index = min(k + timestamp, df_log.shape[0] - 1)
        batch_x = np.expand_dims(df_log.iloc[k:index, :].values, axis=0)
        batch_y = df_log.iloc[k + 1 : index + 1, :].values
        _, loss = sess.run(
            [modelnn.optimizer, modelnn.cost], feed_dict={modelnn.X: batch_x, modelnn.Y: batch_y}
        )
        total_loss += loss
    total_loss /= df_log.shape[0] // timestamp
    if (i + 1) % 100 == 0:
        print("epoch:", i + 1, "avg loss:", total_loss)


# In[8]:


output_predict = np.zeros((df_log.shape[0] + future_day, df_log.shape[1]))
output_predict[0, :] = df_log.iloc[0, :]
upper_b = (df_log.shape[0] // timestamp) * timestamp
for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
    out_logits = sess.run(
        modelnn.logits,
        feed_dict={modelnn.X: np.expand_dims(df_log.iloc[k : k + timestamp, :], axis=0)},
    )
    output_predict[k + 1 : k + timestamp + 1, :] = out_logits

out_logits = sess.run(
    modelnn.logits, feed_dict={modelnn.X: np.expand_dims(df_log.iloc[upper_b:, :], axis=0)}
)
output_predict[upper_b + 1 : df_log.shape[0] + 1, :] = out_logits
df_log.loc[df_log.shape[0]] = out_logits[-1, :]
date_ori.append(date_ori[-1] + timedelta(days=1))


# In[9]:


for i in range(future_day - 1):
    out_logits = sess.run(
        modelnn.logits, feed_dict={modelnn.X: np.expand_dims(df_log.iloc[-timestamp:, :], axis=0)}
    )
    output_predict[df_log.shape[0], :] = out_logits[-1, :]
    df_log.loc[df_log.shape[0]] = out_logits[-1, :]
    date_ori.append(date_ori[-1] + timedelta(days=1))


# In[10]:


df_log = minmax.inverse_transform(output_predict)
date_ori = pd.Series(date_ori).dt.strftime(date_format="%Y-%m-%d").tolist()


# In[11]:


def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer


# In[12]:


current_palette = sns.color_palette("Paired", 12)
fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(111)
x_range_original = np.arange(df.shape[0])
x_range_future = np.arange(df_log.shape[0])
ax.plot(x_range_original, df.iloc[:, 1], label="true Open", color=current_palette[0])
ax.plot(x_range_future, anchor(df_log[:, 0], 0.5), label="predict Open", color=current_palette[1])
ax.plot(x_range_original, df.iloc[:, 2], label="true High", color=current_palette[2])
ax.plot(x_range_future, anchor(df_log[:, 1], 0.5), label="predict High", color=current_palette[3])
ax.plot(x_range_original, df.iloc[:, 3], label="true Low", color=current_palette[4])
ax.plot(x_range_future, anchor(df_log[:, 2], 0.5), label="predict Low", color=current_palette[5])
ax.plot(x_range_original, df.iloc[:, 4], label="true Close", color=current_palette[6])
ax.plot(x_range_future, anchor(df_log[:, 3], 0.5), label="predict Close", color=current_palette[7])
ax.plot(x_range_original, df.iloc[:, 5], label="true Adj Close", color=current_palette[8])
ax.plot(
    x_range_future, anchor(df_log[:, 4], 0.5), label="predict Adj Close", color=current_palette[9]
)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.title("overlap stock market")
plt.xticks(x_range_future[::30], date_ori[::30])
plt.show()


# In[13]:


fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(x_range_original, df.iloc[:, 1], label="true Open", color=current_palette[0])
plt.plot(x_range_original, df.iloc[:, 2], label="true High", color=current_palette[2])
plt.plot(x_range_original, df.iloc[:, 3], label="true Low", color=current_palette[4])
plt.plot(x_range_original, df.iloc[:, 4], label="true Close", color=current_palette[6])
plt.plot(x_range_original, df.iloc[:, 5], label="true Adj Close", color=current_palette[8])
plt.xticks(x_range_original[::60], df.iloc[:, 0].tolist()[::60])
plt.legend()
plt.title("true market")
plt.subplot(1, 2, 2)
plt.plot(x_range_future, anchor(df_log[:, 0], 0.5), label="predict Open", color=current_palette[1])
plt.plot(x_range_future, anchor(df_log[:, 1], 0.5), label="predict High", color=current_palette[3])
plt.plot(x_range_future, anchor(df_log[:, 2], 0.5), label="predict Low", color=current_palette[5])
plt.plot(x_range_future, anchor(df_log[:, 3], 0.5), label="predict Close", color=current_palette[7])
plt.plot(
    x_range_future, anchor(df_log[:, 4], 0.5), label="predict Adj Close", color=current_palette[9]
)
plt.xticks(x_range_future[::60], date_ori[::60])
plt.legend()
plt.title("predict market")
plt.show()


# In[14]:


fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(111)
ax.plot(x_range_original, df.iloc[:, -1], label="true Volume")
ax.plot(x_range_future, anchor(df_log[:, -1], 0.5), label="predict Volume")
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.xticks(x_range_future[::30], date_ori[::30])
plt.title("overlap market volume")
plt.show()


# In[ ]:
