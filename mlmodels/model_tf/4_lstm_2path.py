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


class Model:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias=0.1,
        timestep=5,
        epoch=10,
    ):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)

        self.timestep = timestep
        self.epoch = epoch
        self.hidden_layer_size = num_layers * 2 * size_layer
        with tf.variable_scope("forward", reuse=False):
            rnn_cells_forward = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell(size_layer) for _ in range(num_layers)], state_is_tuple=False
            )
            self.X_forward = tf.placeholder(tf.float32, (None, None, size))
            drop_forward = tf.contrib.rnn.DropoutWrapper(
                rnn_cells_forward, output_keep_prob=forget_bias
            )
            self.hidden_layer_forward = tf.placeholder(tf.float32, (None, self.hidden_layer_size))
            self.outputs_forward, self.last_state_forward = tf.nn.dynamic_rnn(
                drop_forward,
                self.X_forward,
                initial_state=self.hidden_layer_forward,
                dtype=tf.float32,
            )

        with tf.variable_scope("backward", reuse=False):
            rnn_cells_backward = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell(size_layer) for _ in range(num_layers)], state_is_tuple=False
            )
            self.X_backward = tf.placeholder(tf.float32, (None, None, size))
            drop_backward = tf.contrib.rnn.DropoutWrapper(
                rnn_cells_backward, output_keep_prob=forget_bias
            )
            self.hidden_layer_backward = tf.placeholder(tf.float32, (None, self.hidden_layer_size))
            self.outputs_backward, self.last_state_backward = tf.nn.dynamic_rnn(
                drop_backward,
                self.X_backward,
                initial_state=self.hidden_layer_backward,
                dtype=tf.float32,
            )

        self.outputs = self.outputs_backward - self.outputs_forward
        self.Y = tf.placeholder(tf.float32, (None, output_size))
        rnn_W = tf.Variable(tf.random_normal((size_layer, output_size)))
        rnn_B = tf.Variable(tf.random_normal([output_size]))
        self.logits = tf.matmul(self.outputs[-1], rnn_W) + rnn_B
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


def fit(model, data_frame):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(model.epoch):
        init_value_forward = np.zeros((1, model.hidden_layer_size))
        init_value_backward = np.zeros((1, model.hidden_layer_size))
        total_loss = 0
        for k in range(0, data_frame.shape[0] - 1, model.timestep):
            index = min(k + model.timestep, data_frame.shape[0] - 1)
            batch_x_forward = np.expand_dims(data_frame.iloc[k:index, :].values, axis=0)
            batch_x_backward = np.expand_dims(
                np.flip(data_frame.iloc[k:index, :].values, axis=0), axis=0
            )
            batch_y = data_frame.iloc[k + 1 : index + 1, :].values
            last_state_forward, last_state_backward, _, loss = sess.run(
                [model.last_state_forward, model.last_state_backward, model.optimizer, model.cost],
                feed_dict={
                    model.X_forward: batch_x_forward,
                    model.X_backward: batch_x_backward,
                    model.Y: batch_y,
                    model.hidden_layer_forward: init_value_forward,
                    model.hidden_layer_backward: init_value_backward,
                },
            )
            init_value_forward = last_state_forward
            init_value_backward = last_state_backward
            total_loss += loss
        total_loss /= data_frame.shape[0] // model.timestep
        if (i + 1) % 100 == 0:
            print("epoch:", i + 1, "avg loss:", total_loss)
    return sess


def predict(
    model,
    sess,
    data_frame,
    get_hidden_state=False,
    init_value_forward=None,
    init_value_backward=None,
):
    output_predict = np.zeros((data_frame.shape[0], data_frame.shape[1]))
    if init_value_forward is None:
        init_value_forward = np.zeros((1, model.hidden_layer_size))
    if init_value_backward is None:
        init_value_backward = np.zeros((1, model.hidden_layer_size))
    upper_b = (data_frame.shape[0] // model.timestep) * model.timestep

    if upper_b == model.timestep:
        batch_x_forward = np.expand_dims(data_frame.values, axis=0)
        batch_x_backward = np.expand_dims(np.flip(data_frame.values, axis=0), axis=0)
        out_logits, init_value_forward, init_value_backward = sess.run(
            [model.logits, model.last_state_forward, model.last_state_backward],
            feed_dict={
                model.X_forward: batch_x_forward,
                model.X_backward: batch_x_backward,
                model.hidden_layer_forward: init_value_forward,
                model.hidden_layer_backward: init_value_backward,
            },
        )
    else:
        for k in range(0, (data_frame.shape[0] // model.timestep) * model.timestep, model.timestep):

            batch_x_forward = np.expand_dims(
                data_frame.iloc[k : k + model.timestep, :].values, axis=0
            )
            batch_x_backward = np.expand_dims(
                np.flip(data_frame.iloc[k : k + model.timestep, :].values, axis=0), axis=0
            )

            out_logits, init_value_forward, init_value_backward = sess.run(
                [model.logits, model.last_state_forward, model.last_state_backward],
                feed_dict={
                    model.X_forward: batch_x_forward,
                    model.X_backward: batch_x_backward,
                    model.hidden_layer_forward: init_value_forward,
                    model.hidden_layer_backward: init_value_backward,
                },
            )
            output_predict[k + 1 : k + model.timestep + 1, :] = out_logits
    if get_hidden_state:
        return output_predict, init_value_backward, init_value_backward
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
        "4_lstm_2path.py",
        {
            "learning_rate": 0.001,
            "num_layers": 1,
            "size": df_log.shape[1],
            "size_layer": 128,
            "output_size": df_log.shape[1],
            "timestep": 5,
            "epoch": 5,
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

    num_layers = 1
    size_layer = 128
    timestamp = 5
    epoch = 500
    dropout_rate = 0.7
    future_day = 50

    # In[6]:

    tf.reset_default_graph()
    modelnn = Model(
        0.01,
        num_layers,
        df_log.shape[1],
        size_layer,
        df_log.shape[1],
        dropout_rate,
        timestamp,
        epoch,
    )
    sess = fit(modelnn, df_log)
    # In[8]:

    output_predict = np.zeros((df_log.shape[0] + future_day, df_log.shape[1]))
    output_predict[0, :] = df_log.iloc[0, :]
    upper_b = (df_log.shape[0] // timestamp) * timestamp

    output_predict[: df_log.shape[0], :], init_value_forward, init_value_backward = predict(
        modelnn, sess, df_log, True
    )

    out_logits, init_value_forward, init_value_backward = predict(
        modelnn, sess, df_log.iloc[upper_b:, :], True, init_value_forward, init_value_backward
    )

    output_predict[upper_b + 1 : df_log.shape[0] + 1, :] = out_logits
    df_log.loc[df_log.shape[0]] = out_logits[-1, :]
    date_ori.append(date_ori[-1] + timedelta(days=1))

    # In[9]:

    for i in range(future_day - 1):
        out_logits, init_value_forward, init_value_backward = predict(
            modelnn,
            sess,
            df_log.iloc[-timestamp:, :],
            True,
            init_value_forward,
            init_value_backward,
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
