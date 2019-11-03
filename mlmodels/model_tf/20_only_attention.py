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


# In[5]:


def sinusoidal_positional_encoding(inputs, num_units, zero_pad=False, scale=False):
    T = inputs.get_shape().as_list()[1]
    position_idx = tf.tile(tf.expand_dims(tf.range(T), 0), [tf.shape(inputs)[0], 1])
    position_enc = np.array(
        [[pos / np.power(10000, 2.0 * i / num_units) for i in range(num_units)] for pos in range(T)]
    )
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
    lookup_table = tf.convert_to_tensor(position_enc, tf.float32)
    if zero_pad:
        lookup_table = tf.concat([tf.zeros([1, num_units]), lookup_table[1:, :]], axis=0)
    outputs = tf.nn.embedding_lookup(lookup_table, position_idx)
    if scale:
        outputs = outputs * num_units ** 0.5
    return outputs


class Model:
    def __init__(self, seq_len, learning_rate, dimension_input, dimension_output, epoch=100):
        self.epoch = epoch
        self.timestep = seq_len
        self.X = tf.placeholder(tf.float32, [None, seq_len, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        x = self.X
        x += sinusoidal_positional_encoding(x, dimension_input)
        masks = tf.sign(self.X[:, :, 0])
        align = tf.squeeze(tf.layers.dense(x, 1, tf.tanh), -1)
        paddings = tf.fill(tf.shape(align), float("-inf"))
        align = tf.where(tf.equal(masks, 0), paddings, align)
        align = tf.expand_dims(tf.nn.softmax(align), -1)
        x = tf.squeeze(tf.matmul(tf.transpose(x, [0, 2, 1]), align), -1)
        self.logits = tf.layers.dense(x, dimension_output)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


def fit(model, data_frame):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(model.epoch):
        total_loss = 0
        for k in range(0, (data_frame.shape[0] // model.timestep) * model.timestep, model.timestep):
            batch_x = np.expand_dims(data_frame.iloc[k : k + model.timestep].values, axis=0)
            batch_y = data_frame.iloc[k + 1 : k + model.timestep + 1].values
            _, loss = sess.run(
                [model.optimizer, model.cost], feed_dict={model.X: batch_x, model.Y: batch_y}
            )
            loss = np.mean(loss)
            total_loss += loss
        total_loss /= data_frame.shape[0] // model.timestep
        if (i + 1) % 100 == 0:
            print("epoch:", i + 1, "avg loss:", total_loss)
    return sess


def predict(model, sess, data_frame):
    output_predict = np.zeros((data_frame.shape[0], data_frame.shape[1]))
    upper_b = (data_frame.shape[0] // model.timestep) * model.timestep

    if upper_b == model.timestep:
        out_logits = sess.run(
            model.logits, feed_dict={model.X: np.expand_dims(data_frame.values, axis=0)}
        )
    else:
        for k in range(0, (data_frame.shape[0] // model.timestep) * model.timestep, model.timestep):
            out_logits = sess.run(
                model.logits,
                feed_dict={
                    model.X: np.expand_dims(data_frame.iloc[k : k + model.timestep], axis=0)
                },
            )
            output_predict[k + 1 : k + model.timestep + 1] = out_logits
    return output_predict


def test(filename="dataset/GOOG-year.csv"):
    import os, sys, inspect

    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from models import create, fit, predict

    df = pd.read_csv(filename)
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
    print(df.head(5))

    minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype("float32"))
    df_log = minmax.transform(df.iloc[:, 1:].astype("float32"))
    df_log = pd.DataFrame(df_log)

    module, model = create(
        "20_only_attention.py",
        {
            "epoch": 1,
            "seq_len": 5,
            "learning_rate": 0.01,
            "dimension_input": df_log.shape[1],
            "dimension_output": df_log.shape[1],
        },
    )

    sess = fit(model, module, df_log)
    predictions = predict(model, module, sess, df_log)
    print(predictions)


if __name__ == "__main__":
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

    timestamp = 5
    epoch = 500
    future_day = 50

    # In[6]:

    tf.reset_default_graph()
    modelnn = Model(timestamp, 0.01, df_log.shape[1], df_log.shape[1], epoch=epoch)

    sess = fit(modelnn, df_log)

    # In[8]:

    output_predict = np.zeros((df_log.shape[0] + future_day, df_log.shape[1]))
    output_predict[0] = df_log.iloc[0]
    upper_b = (df_log.shape[0] // timestamp) * timestamp

    output_predict[: df_log.shape[0], :] = predict(modelnn, sess, df_log)
    output_predict[upper_b + 1 : df_log.shape[0] + 1] = predict(
        modelnn, sess, df_log.iloc[upper_b:]
    )

    df_log.loc[df_log.shape[0]] = output_predict[upper_b + 1 : df_log.shape[0] + 1][-1]
    date_ori.append(date_ori[-1] + timedelta(days=1))

    # In[9]:

    for i in range(future_day - 1):
        out_logits = predict(modelnn, sess, df_log.iloc[-timestamp:])
        output_predict[df_log.shape[0]] = out_logits[-1]
        df_log.loc[df_log.shape[0]] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(days=1))

    # In[10]:

    df_log = minmax.inverse_transform(df_log.values)
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
    ax.plot(
        x_range_future, anchor(df_log[:, 0], 0.5), label="predict Open", color=current_palette[1]
    )
    ax.plot(x_range_original, df.iloc[:, 2], label="true High", color=current_palette[2])
    ax.plot(
        x_range_future, anchor(df_log[:, 1], 0.5), label="predict High", color=current_palette[3]
    )
    ax.plot(x_range_original, df.iloc[:, 3], label="true Low", color=current_palette[4])
    ax.plot(
        x_range_future, anchor(df_log[:, 2], 0.5), label="predict Low", color=current_palette[5]
    )
    ax.plot(x_range_original, df.iloc[:, 4], label="true Close", color=current_palette[6])
    ax.plot(
        x_range_future, anchor(df_log[:, 3], 0.5), label="predict Close", color=current_palette[7]
    )
    ax.plot(x_range_original, df.iloc[:, 5], label="true Adj Close", color=current_palette[8])
    ax.plot(
        x_range_future,
        anchor(df_log[:, 4], 0.5),
        label="predict Adj Close",
        color=current_palette[9],
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
    plt.plot(
        x_range_future, anchor(df_log[:, 0], 0.5), label="predict Open", color=current_palette[1]
    )
    plt.plot(
        x_range_future, anchor(df_log[:, 1], 0.5), label="predict High", color=current_palette[3]
    )
    plt.plot(
        x_range_future, anchor(df_log[:, 2], 0.5), label="predict Low", color=current_palette[5]
    )
    plt.plot(
        x_range_future, anchor(df_log[:, 3], 0.5), label="predict Close", color=current_palette[7]
    )
    plt.plot(
        x_range_future,
        anchor(df_log[:, 4], 0.5),
        label="predict Adj Close",
        color=current_palette[9],
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

    # In[15]:

    fig = plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(x_range_original, df.iloc[:, -1], label="true Volume")
    plt.xticks(x_range_original[::60], df.iloc[:, 0].tolist()[::60])
    plt.legend()
    plt.title("true market volume")
    plt.subplot(1, 2, 2)
    plt.plot(x_range_future, anchor(df_log[:, -1], 0.5), label="predict Volume")
    plt.xticks(x_range_future[::60], date_ori[::60])
    plt.legend()
    plt.title("predict market volume")
    plt.show()

    # In[ ]:
