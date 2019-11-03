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
from sklearn.preprocessing import MinMaxScaler

sns.set()


# In[2]:


class Model:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias=0.1,
        epoch=500,
        timestep=5,
    ):

        self.num_layers = num_layers
        self.forget_bias = forget_bias
        self.size = size
        self.size_layer = size_layer
        self.hidden_layer_size = num_layers * size_layer
        self.output_size = output_size
        self.timestep = timestep
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.build_model()

    def build_model(self):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.BasicRNNCell(size_layer)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(self.size_layer) for _ in range(self.num_layers)], state_is_tuple=False
        )
        self.X = tf.placeholder(tf.float32, (None, None, self.size))
        self.Y = tf.placeholder(tf.float32, (None, self.output_size))
        drop = tf.contrib.rnn.DropoutWrapper(rnn_cells, output_keep_prob=self.forget_bias)
        self.hidden_layer = tf.placeholder(tf.float32, (None, self.hidden_layer_size))
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state=self.hidden_layer, dtype=tf.float32
        )
        rnn_W = tf.Variable(tf.random_normal((self.size_layer, self.output_size)))
        rnn_B = tf.Variable(tf.random_normal([self.output_size]))
        self.logits = tf.matmul(self.outputs[-1], rnn_W) + rnn_B
        self.logits = tf.layers.dense(self.outputs[-1], self.output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)


# In[5]:


def reducedimension(input_, dimension=2, learning_rate=0.01, hidden_layer=256, epoch=20):

    input_size = input_.shape[1]
    X = tf.placeholder("float", [None, input_size])
    first_layer_encoder = tf.layers.dense(X, hidden_layer, activation=tf.nn.sigmoid)
    second_layer_encoder = tf.layers.dense(first_layer_encoder, dimension, activation=tf.nn.sigmoid)
    first_layer_decoder = tf.layers.dense(
        second_layer_encoder, hidden_layer, activation=tf.nn.sigmoid
    )
    second_layer_decoder = tf.layers.dense(
        first_layer_decoder, input_size, activation=tf.nn.sigmoid
    )
    cost = tf.reduce_mean(tf.square(X - second_layer_decoder))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        last_time = time.time()
        _, loss = sess.run([optimizer, cost], feed_dict={X: input_})
        if (i + 1) % 10 == 0:
            print("epoch:", i + 1, "loss:", loss, "time:", time.time() - last_time)

    vectors = sess.run(second_layer_encoder, feed_dict={X: input_})
    return vectors, sess, second_layer_encoder, X


def fit(model, data_frame):
    tf.reset_default_graph()
    thought_vector, sess_reduction, second_layer_encoder, X = reducedimension(
        data_frame.values,
        dimension=16,
        learning_rate=model.learning_rate,
        hidden_layer=model.size_layer,
        epoch=100,
    )
    tf.reset_default_graph()
    model.build_model()  # to put the model in the graph

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(model.epoch):
        init_value = np.zeros((1, model.hidden_layer_size))
        total_loss = 0
        for k in range(0, thought_vector.shape[0] - 1, model.timestep):
            index = min(k + model.timestep, thought_vector.shape[0] - 1)
            batch_x = np.expand_dims(thought_vector[k:index, :], axis=0)
            batch_y = data_frame.iloc[k + 1 : index + 1, :].values
            init_value, _, loss = sess.run(
                [model.last_state, model.optimizer, model.cost],
                feed_dict={model.X: batch_x, model.Y: batch_y, model.hidden_layer: init_value},
            )
            loss = np.mean(loss)
            total_loss += loss
        total_loss /= data_frame.shape[0] // model.timestep
        if (i + 1) % 100 == 0:
            print("epoch:", i + 1, "avg loss:", total_loss)
    return {
        "main": sess,
        "reduction": sess_reduction,
        "second_layer_encoder": second_layer_encoder,
        "X": X,
    }


def predict(model, sess, data_frame):
    thought_vector = sess["reduction"].run(
        sess["second_layer_encoder"], feed_dict={sess["X"]: data_frame.values}
    )
    sess = sess["main"]
    output_predict = np.zeros((data_frame.shape[0] + 1, data_frame.shape[1]))
    output_predict[0, :] = data_frame.iloc[0, :]
    upper_b = (data_frame.shape[0] // model.timestep) * model.timestep
    init_value = np.zeros((1, model.hidden_layer_size))
    for k in range(0, (data_frame.shape[0] // model.timestep) * model.timestep, model.timestep):
        out_logits, last_state = sess.run(
            [model.logits, model.last_state],
            feed_dict={
                model.X: np.expand_dims(thought_vector[k : k + model.timestep], axis=0),
                model.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[k + 1 : k + model.timestep + 1] = out_logits

    out_logits, last_state = sess.run(
        [model.logits, model.last_state],
        feed_dict={
            model.X: np.expand_dims(thought_vector[upper_b:], axis=0),
            model.hidden_layer: init_value,
        },
    )
    init_value = last_state
    output_predict[upper_b + 1 : data_frame.shape[0] + 1] = out_logits
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
        "10_encoder_vanilla.py",
        {
            "learning_rate": 0.001,
            "num_layers": 1,
            "size": 16,
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

    # In[3]:

    df = pd.read_csv("../dataset/GOOG-year.csv")
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
    df.head()

    # In[4]:

    minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype("float32"))
    df_log = minmax.transform(df.iloc[:, 1:].astype("float32"))
    df_log = pd.DataFrame(df_log)
    df_log.head()

    # In[8]:

    num_layers = 1
    size_layer = 128
    timestamp = 5
    epoch = 500
    dropout_rate = 0.1

    # In[9]:

    tf.reset_default_graph()
    modelnn = Model(
        0.001, num_layers, 16, size_layer, df_log.shape[1], dropout_rate, epoch, timestamp
    )

    sess = fit(modelnn, df_log)

    # In[11]:

    output_predict = predict(modelnn, sess, df_log)
    df_log.loc[df_log.shape[0]] = output_predict[-1]
    date_ori.append(date_ori[-1] + timedelta(days=1))

    # In[12]:

    df_log = minmax.inverse_transform(output_predict)
    date_ori = pd.Series(date_ori).dt.strftime(date_format="%Y-%m-%d").tolist()

    # In[13]:

    def anchor(signal, weight):
        buffer = []
        last = signal[0]
        for i in signal:
            smoothed_val = last * weight + (1 - weight) * i
            buffer.append(smoothed_val)
            last = smoothed_val
        return buffer

    # In[14]:

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

    # In[ ]:
