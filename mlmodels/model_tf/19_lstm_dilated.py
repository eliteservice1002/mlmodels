#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

sns.set()


# In[5]:


def contruct_cells(hidden_structs):
    cells = []
    for hidden_dims in hidden_structs:
        cells.append(
            tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(hidden_dims, state_is_tuple=False), output_keep_prob=0.7
            )
        )
    return cells


def rnn_reformat(x, input_dims, n_steps):
    x_ = tf.transpose(x, [1, 0, 2])
    x_ = tf.reshape(x_, [-1, input_dims])
    return tf.split(x_, n_steps, 0)


def dilated_rnn(cell, inputs, rate, states, scope="default"):
    n_steps = len(inputs)
    if not (n_steps % rate) == 0:
        zero_tensor = tf.zeros_like(inputs[0])
        dilated_n_steps = n_steps // rate + 1
        for i_pad in range(dilated_n_steps * rate - n_steps):
            inputs.append(zero_tensor)
    else:
        dilated_n_steps = n_steps // rate
    dilated_inputs = [
        tf.concat(inputs[i * rate : (i + 1) * rate], axis=0) for i in range(dilated_n_steps)
    ]
    dilated_outputs, states = tf.contrib.rnn.static_rnn(
        cell, dilated_inputs, initial_state=states, dtype=tf.float32, scope=scope
    )
    splitted_outputs = [tf.split(output, rate, axis=0) for output in dilated_outputs]
    unrolled_outputs = [output for sublist in splitted_outputs for output in sublist]
    return unrolled_outputs[:n_steps], states


def multi_dilated_rnn(cells, inputs, dilations, states):
    x = copy.copy(inputs)
    count = 0
    for cell, dilation in zip(cells, dilations):
        x, states = dilated_rnn(cell, x, dilation, states, scope="multi_dilated_rnn_%d" % count)
        count += 1
    return x, states


class Model:
    def __init__(
        self,
        steps,
        dimension_input,
        dimension_output,
        learning_rate=0.001,
        hidden_structs=[20],
        dilations=[1, 1, 1, 1],
        epoch=500,
    ):
        self.epoch = epoch
        self.timestep = steps
        self.hidden_layer_size = 40
        hidden_structs = hidden_structs * len(dilations)
        self.X = tf.placeholder(tf.float32, [None, steps, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        self.hidden_layer = tf.placeholder(tf.float32, (None, 40))
        x_reformat = rnn_reformat(self.X, dimension_input, steps)
        cells = contruct_cells(hidden_structs)
        layer_outputs, self.last_state = multi_dilated_rnn(
            cells, x_reformat, dilations, self.hidden_layer
        )
        if dilations[0] == 1:
            weights = tf.Variable(tf.random_normal(shape=[hidden_structs[-1], dimension_output]))
            bias = tf.Variable(tf.random_normal(shape=[dimension_output]))
            self.logits = tf.matmul(layer_outputs[-1], weights) + bias
        else:
            weights = tf.Variable(
                tf.random_normal(shape=[hidden_structs[-1] * dilations[0], dimension_output])
            )
            bias = tf.Variable(tf.random_normal(shape=[dimension_output]))
            for idx, i in enumerate(range(-dilations[0], 0, 1)):
                if idx == 0:
                    hidden_outputs_ = layer_outputs[i]
                else:
                    hidden_outputs_ = tf.concat([hidden_outputs_, layer_outputs[i]], axis=1)
            self.logits = tf.matmul(hidden_outputs_, weights) + bias
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


def fit(model, data_frame):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(model.epoch):

        init_value = np.zeros((1, model.hidden_layer_size))
        total_loss = 0
        for k in range(0, (data_frame.shape[0] // model.timestep) * model.timestep, model.timestep):
            batch_x = np.expand_dims(data_frame.iloc[k : k + model.timestep].values, axis=0)
            batch_y = data_frame.iloc[k + 1 : k + model.timestep + 1].values
            last_state, _, loss = sess.run(
                [model.last_state, model.optimizer, model.cost],
                feed_dict={model.X: batch_x, model.Y: batch_y, model.hidden_layer: init_value},
            )
            init_value = last_state
            total_loss += loss
        total_loss /= data_frame.shape[0] // model.timestep
        if (i + 1) % 100 == 0:
            print("epoch:", i + 1, "avg loss:", total_loss)
    return sess


def predict(model, sess, data_frame, get_hidden_state=False, init_value=None):
    if init_value is None:
        init_value = np.zeros((1, model.hidden_layer_size))
    output_predict = np.zeros((data_frame.shape[0], data_frame.shape[1]))
    upper_b = (data_frame.shape[0] // model.timestep) * model.timestep

    if upper_b == model.timestep:
        out_logits, init_value = sess.run(
            [model.logits, model.last_state],
            feed_dict={
                model.X: np.expand_dims(data_frame.values, axis=0),
                model.hidden_layer: init_value,
            },
        )
    else:
        for k in range(0, (data_frame.shape[0] // model.timestep) * model.timestep, model.timestep):
            out_logits, last_state = sess.run(
                [model.logits, model.last_state],
                feed_dict={
                    model.X: np.expand_dims(data_frame.iloc[k : k + model.timestep].values, axis=0),
                    model.hidden_layer: init_value,
                },
            )
            init_value = last_state
            output_predict[k + 1 : k + model.timestep + 1] = out_logits
    if get_hidden_state:
        return output_predict, init_value
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
        "19_lstm_dilated",
        {
            "steps": 5,
            "dimension_input": df_log.shape[1],
            "dimension_output": df_log.shape[1],
            "epoch": 1,
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
    dropout_rate = 0.7
    future_day = 50

    # In[6]:

    tf.reset_default_graph()
    modelnn = model = Model(timestamp, df_log.shape[1], df_log.shape[1], epoch=epoch)
    sess = fit(modelnn, df_log)

    # In[8]:

    output_predict = np.zeros((df_log.shape[0] + future_day, df_log.shape[1]))
    output_predict[0] = df_log.iloc[0]
    upper_b = (df_log.shape[0] // timestamp) * timestamp
    output_predict[: df_log.shape[0], :], init_value = predict(modelnn, sess, df_log, True)

    output_predict[upper_b + 1 : df_log.shape[0] + 1], init_value = predict(
        modelnn, sess, df_log.iloc[upper_b:], True, init_value
    )
    df_log.loc[df_log.shape[0]] = output_predict[upper_b + 1 : df_log.shape[0] + 1][-1]

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
