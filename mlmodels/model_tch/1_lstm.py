# coding: utf-8
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch as tch


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
        epoch=5,
    ):
        self.epoch = epoch
        self.timestep = timestep
        self.hidden_layer_size = num_layers * 2 * size_layer


def fit(model, df):
    """
      data_params : dateframe containing the training data
    
    """
    for i in range(model.epoch):

        init_value = np.zeros((1, model.hidden_layer_size))
        total_loss = 0
        for k in range(0, df.shape[0] - 1, model.timestep):

            init_value = last_state
            total_loss += loss
        total_loss /= df.shape[0] // model.timestep
        if (i + 1) % 100 == 0:
            print("epoch:", i + 1, "avg loss:", total_loss)
    return sess


def predict(model, df, get_hidden_state=False, init_value=None):
    pass


def params(choice=None):
    # output parms
    if choice is None:
        return {
            "learning_rate": 0.001,
            "num_layers": 1,
            "size": df_log.shape[1],
            "size_layer": 128,
            "output_size": df_log.shape[1],
            "timestep": 5,
            "epoch": 5,
        }


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
        "model_tch/1_lstm",
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


####################################################################################################
####################################################################################################
if __name__ == "__main__":
    num_layers = 1
    size_layer = 128
    timestamp = 5
    epoch = 5
    dropout_rate = 0.7
    future_day = 50

    # In[2]:
    df = pd.read_csv("../dataset/GOOG-year.csv")
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
    df.head()

    # In[3]:
    minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype("float32"))
    df_log = minmax.transform(df.iloc[:, 1:].astype("float32"))
    df_log = pd.DataFrame(df_log)
    df_log.head()

    pass
