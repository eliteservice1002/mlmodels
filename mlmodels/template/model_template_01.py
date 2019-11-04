# coding: utf-8
"""
LSTM Time series predictions
python  model_tf/1_lstm.py


"""
import os, sys, inspect

import numpy as np
import pandas as pd



import tensorflow as tf


# from util import set_root_dir
####################################################################################################
class Model:
    def __init__(self,
        learning_rate=0.001,
        num_layers=2,
        size=None,
        size_layer=128,
        output_size=None,
        forget_bias=0.1,
        timestep=5,
        epoch=5,
    ):
        self.epoch = epoch
        self.stats = {"loss":0.0,
                      "loss_history": [] }
        self.timestep = timestep







        self.logits = tf.layers.dense(self.outputs[-1], output_size)

        ### Regression loss
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)






def fit(model, data_params):
    df = get_dataset(data_params)

    #########
    nlog_freq=100
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(model.epoch):
        total_loss = 0


        ######## Model specific  ########################################






        ####### End Model specific    ##################################


        total_loss /= df.shape[0] // model.timestep
        model.stats["loss"] = total_loss

        if (i + 1) % nlog_freq == 0:
            print("epoch:", i + 1, "avg loss:", total_loss)
    return sess



def stats_compute(model, sess, df, ):
    # Compute stats on training
    arr_out = predict(model, sess, df, )
    return model.stats



def predict(model, sess, data_params, ):

    df = get_dataset(data_params)


    return output_predict



def reset_model():
    tf.reset_default_graph()



####################################################################################################
def set_root_dir():
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    return parent_dir



def get_dataset(data_params=None):
    """
      JSON data_params  to  actual dataframe of data
    """

    filename = data_params["data_path"]  #

    ##### Specific   ######################################################
    set_root_dir()

    df = pd.read_csv(filename)


    return df


def get_params(choice="test", **kwargs):
    # output parms
    # print(kwargs)
    if choice=="test":
        p=         { "learning_rate": 0.001,
            "num_layers": 1,
            "size": None,
            "size_layer": 128,
            "output_size": None,
            "timestep": 4,
            "epoch": 2,
        }

        ### Overwrite by manual input
        for k,x in kwargs.items() :
            p[k] = x

        return p



####################################################################################################
def test(data_path="dataset/GOOG-year.csv", reset=True):
    set_root_dir()
    data_params = { "data_path" : data_path, "data_type" : "pandas" }

    df = get_dataset(data_params)
    model_params = get_params("test", size=df.shape[1], output_size=df.shape[1]  )


    from models import create_full, fit, predict
    module, model = create_full( "model_tf.1_lstm", model_params)

    sess = fit(model, module, data_params)
    predictions = predict(model, sess, data_params)
    print(predictions)
    tf.reset_default_graph()



def test2( data_path="dataset/GOOG-year.csv" ):
    data_params = { "data_path" : data_path, "data_type" : "pandas" }

    df = get_dataset(data_params)
    model_params = get_params("test", size=df.shape[1], output_size=df.shape[1]  )


    model = Model(**model_params)
    sess  = fit(model, data_params)
    predictions = predict(model, sess, data_params)
    print(predictions)
    tf.reset_default_graph()



####################################################################################################
####################################################################################################
if __name__ == "__main__":
    test2()







