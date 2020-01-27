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


def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable("gamma", params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable("beta", params_shape, tf.float32, tf.zeros_initializer())

    outputs = gamma * normalized + beta
    return outputs


def multihead_attn(queries, keys, q_masks, k_masks, future_binding, num_units, num_heads):

    T_q = tf.shape(queries)[1]
    T_k = tf.shape(keys)[1]

    Q = tf.layers.dense(queries, num_units, name="Q")
    K_V = tf.layers.dense(keys, 2 * num_units, name="K_V")
    K, V = tf.split(K_V, 2, -1)

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

    align = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
    align = align / np.sqrt(K_.get_shape().as_list()[-1])

    paddings = tf.fill(tf.shape(align), float("-inf"))

    key_masks = k_masks
    key_masks = tf.tile(key_masks, [num_heads, 1])
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, T_q, 1])
    align = tf.where(tf.equal(key_masks, 0), paddings, align)

    if future_binding:
        lower_tri = tf.ones([T_q, T_k])
        lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()
        masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
        align = tf.where(tf.equal(masks, 0), paddings, align)

    align = tf.nn.softmax(align)
    query_masks = tf.to_float(q_masks)
    query_masks = tf.tile(query_masks, [num_heads, 1])
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, T_k])
    align *= query_masks

    outputs = tf.matmul(align, V_)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
    outputs += queries
    outputs = layer_norm(outputs)
    return outputs


def pointwise_feedforward(inputs, hidden_units, activation=None):
    outputs = tf.layers.dense(inputs, 4 * hidden_units, activation=activation)
    outputs = tf.layers.dense(outputs, hidden_units, activation=None)
    outputs += inputs
    outputs = layer_norm(outputs)
    return outputs


def learned_position_encoding(inputs, mask, embed_dim):
    T = tf.shape(inputs)[1]
    outputs = tf.range(tf.shape(inputs)[1])  # (T_q)
    outputs = tf.expand_dims(outputs, 0)  # (1, T_q)
    outputs = tf.tile(outputs, [tf.shape(inputs)[0], 1])  # (N, T_q)
    outputs = embed_seq(outputs, T, embed_dim, zero_pad=False, scale=False)
    return tf.expand_dims(tf.to_float(mask), -1) * outputs


def sinusoidal_position_encoding(inputs, mask, repr_dim):
    T = tf.shape(inputs)[1]
    pos = tf.reshape(tf.range(0.0, tf.to_float(T), dtype=tf.float32), [-1, 1])
    i = np.arange(0, repr_dim, 2, np.float32)
    denom = np.reshape(np.power(10000.0, i / repr_dim), [1, -1])
    enc = tf.expand_dims(tf.concat([tf.sin(pos / denom), tf.cos(pos / denom)], 1), 0)
    return tf.tile(enc, [tf.shape(inputs)[0], 1, 1]) * tf.expand_dims(tf.to_float(mask), -1)


def label_smoothing(inputs, epsilon=0.1):
    C = inputs.get_shape().as_list()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / C)


class Attention:
    def __init__(
        self,
        size_layer,
        embedded_size,
        learning_rate,
        size,
        output_size,
        num_blocks=2,
        num_heads=8,
        min_freq=50,
    ):
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))

        encoder_embedded = tf.layers.dense(self.X, embedded_size)
        encoder_embedded = tf.nn.dropout(encoder_embedded, keep_prob=0.75)
        x_mean = tf.reduce_mean(self.X, axis=2)
        en_masks = tf.sign(x_mean)
        encoder_embedded += sinusoidal_position_encoding(self.X, en_masks, embedded_size)

        for i in range(num_blocks):
            with tf.variable_scope("encoder_self_attn_%d" % i, reuse=tf.AUTO_REUSE):
                encoder_embedded = multihead_attn(
                    queries=encoder_embedded,
                    keys=encoder_embedded,
                    q_masks=en_masks,
                    k_masks=en_masks,
                    future_binding=False,
                    num_units=size_layer,
                    num_heads=num_heads,
                )

            with tf.variable_scope("encoder_feedforward_%d" % i, reuse=tf.AUTO_REUSE):
                encoder_embedded = pointwise_feedforward(
                    encoder_embedded, embedded_size, activation=tf.nn.relu
                )
        self.logits = tf.layers.dense(encoder_embedded[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


# In[5]:


timestamp = 5
epoch = 500
future_day = 50
embedded_size = 128
learning_rate = 0.001


# In[6]:


tf.reset_default_graph()
modelnn = Attention(embedded_size, embedded_size, learning_rate, df_log.shape[1], df_log.shape[1])
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
