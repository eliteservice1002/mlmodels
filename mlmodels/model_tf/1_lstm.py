# coding: utf-8
"""
LSTM Time series predictions
python  model_tf/1_lstm.py


"""
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)


import os, sys, inspect
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



import tensorflow as tf
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # **** change the warning level ****


####################################################################################################
from mlmodels.model_tf.util import os_package_root_path, os_file_path, os_module_path


####################################################################################################
def log(*s, n=0,m=1):
  sspace = "#" * n
  sjump =  "\n" * m
  print(sjump, sspace, s, sspace, flush=True)



####################################################################################################
class Model:
    def __init__(self,
        epoch=5,
        learning_rate=0.001,
        
        num_layers=2,
        size=None,
        size_layer=128,
        output_size=None,
        forget_bias=0.1,
        timestep=5,
    ):
        self.epoch = epoch
        self.stats = {"loss":0.0,
                      "loss_history": [] }

        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))

        ### Model Structure        ################################
        self.timestep = timestep
        self.hidden_layer_size = num_layers * 2 * size_layer

        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)], state_is_tuple=False
        )
        
        drop = tf.contrib.rnn.DropoutWrapper(rnn_cells, output_keep_prob=forget_bias)
        
        self.hidden_layer = tf.placeholder(tf.float32, (None, self.hidden_layer_size))
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state=self.hidden_layer, dtype=tf.float32
        )
        self.logits = tf.layers.dense(self.outputs[-1], output_size)


        ### Loss    ##############################################
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self.cost)



def fit(model, data_pars, compute_pars={}, out_pars=None,  **kwargs):
    df = get_dataset(data_pars)
    print(df.head(5))
    msample = df.shape[0]
    nlog_freq = compute_pars.get("nlog_freq", 100)
    
    
    ######################################################################
    sess =   tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(model.epoch):
        total_loss = 0.0


        ######## Model specific  ########################################
        init_value = np.zeros((1, model.hidden_layer_size))
        for k in range(0, df.shape[0] - 1, model.timestep):
            index = min(k + model.timestep, df.shape[0] - 1)
            batch_x = np.expand_dims(df.iloc[k:index, :].values, axis=0)
            batch_y = df.iloc[k + 1 : index + 1, :].values
            last_state, _, loss = sess.run(
                [model.last_state, model.optimizer, model.cost],
                feed_dict={model.X: batch_x, model.Y: batch_y, model.hidden_layer: init_value},
            )
            init_value = last_state
            total_loss += loss
        ####### End Model specific    ##################################


        total_loss /= msample // model.timestep
        model.stats["loss"] = total_loss

        if (i + 1) % nlog_freq == 0:
            print("epoch:", i + 1, "avg loss:", total_loss)
    return sess



def metrics(model, sess=None, data_pars=None, compute_pars=None, out_pars=None):
    """
       Return metrics of the model stored
    #### SK-Learn metrics
    # Compute stats on training
    #df = get_dataset(data_pars)
    #arr_out = predict(model, sess, df, get_hidden_state=False, init_value=None)
    :param model:
    :param sess:
    :param data_pars:
    :param out_pars:
    :return:
    """

    return model.stats



def predict(model, sess, data_pars=None,  out_pars=None,   compute_pars=None,
            get_hidden_state=False, init_value=None):
    df = get_dataset(data_pars)
    print(df, flush=True)


    #############################################################
    if init_value is None:
        init_value = np.zeros((1, model.hidden_layer_size))
    output_predict = np.zeros((df.shape[0], df.shape[1]))
    upper_b = (df.shape[0] // model.timestep) * model.timestep


    if upper_b == model.timestep:
        out_logits, init_value = sess.run(
            [model.logits, model.last_state],
            feed_dict={
                model.X: np.expand_dims(df.values, axis=0),
                model.hidden_layer: init_value,
            },
        )
        output_predict[ 1:  model.timestep + 1] = out_logits
        
    else:
        for k in range(0, (df.shape[0] // model.timestep) * model.timestep, model.timestep):
            out_logits, last_state = sess.run(
                [model.logits, model.last_state],
                feed_dict={ model.X: np.expand_dims(df.iloc[k : k + model.timestep].values, axis=0),
                            model.hidden_layer: init_value,
                },
            )
            init_value = last_state
            output_predict[k + 1 : k + model.timestep + 1] = out_logits
    
    
    if get_hidden_state:
        return output_predict, init_value
    return output_predict



def reset_model():
    tf.compat.v1.reset_default_graph()



####################################################################################################
def get_dataset(data_pars=None):
    """
      JSON data_pars  to  actual dataframe of data
    """
    print(data_pars)
    filename = data_pars["data_path"]  #
    

    ##### Specific   ######################################################
    df = pd.read_csv(filename)
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
    print( filename )
    print(df.head(5))

    minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype("float32"))
    df_log = minmax.transform(df.iloc[:, 1:].astype("float32"))
    df_log = pd.DataFrame(df_log)
    return df_log


def get_pars(choice="test", **kwargs):
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
def test(data_path="dataset/GOOG-year.csv", out_path="", reset=True):
    """
       Using mlmodels package method
       path : mlmodels/mlmodels/dataset/
       from ../../model_tf
      
    """
    data_path =    os_package_root_path(__file__, sublevel=1, path_add=data_path)
    print(data_path)
    
    log("############# Data, Params preparation   #################")
    data_pars = { "data_path" :  data_path, "data_type" : "pandas" }
    out_pars = { "path" : data_path +  out_path }
    compute_pars = {}

    df = get_dataset(data_pars)
    model_pars = get_pars("test", size=df.shape[1], output_size=df.shape[1] )


    log("############ Model preparation   #########################")
    from mlmodels.models import module_load_full, fit, predict
    module, model = module_load_full( "model_tf.1_lstm", model_pars)
    print(module, model)


    log("############ Model fit   ##################################")
    sess = fit(model, module, data_pars=data_pars, out_pars=out_pars, compute_pars={})
    print("fit success", sess)
    
    log("############ Prediction##########################")
    preds = predict(model, module, sess, data_pars=data_pars,
                    out_pars= out_pars, compute_pars= compute_pars)
    print(preds)
    


def test2( data_path="dataset/GOOG-year.csv" ):
    """
      Using this file methods
    """
    #### path to local package roots
    data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
    print(data_path)
    
    data_pars = { "data_path" : data_path, "data_type" : "pandas" }
    out_pars = { "path" : data_path  }
    compute_pars = {}


    ###Need to get variable size to initiatlize the model
    df = get_dataset(data_pars)
    model_pars = get_pars("test", size=df.shape[1], output_size=df.shape[1] )


    #### Model setup, fit, predict
    model = Model(**model_pars)
    sess  = fit(model, data_path=data_pars)
    predictions = predict(model, sess, data_pars)
    print(predictions)




if __name__ == "__main__":
    test2()
















####################################################################################################
####################################################################################################

    """
    import seaborn as sns
    sns.set()
    import matplotlib.pyplot as plt

    num_layers = 1
    size_layer = 128
    timestamp = 5
    epoch = 5
    dropout_rate = 0.7
    future_day = 50

    # In[2]:
    data_pars = pd.read_csv("../dataset/GOOG-year.csv")
    date_ori = pd.to_datetime(data_pars.iloc[:, 0]).tolist()
    data_pars.head()

    # In[3]:
    minmax = MinMaxScaler().fit(data_pars.iloc[:, 1:].astype("float32"))
    df_log = minmax.transform(data_pars.iloc[:, 1:].astype("float32"))
    df_log = pd.DataFrame(df_log)
    df_log.head()

    # In[4]:
    # In[7]:

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

    # In[9]:

    output_predict = np.zeros((df_log.shape[0] + future_day, df_log.shape[1]))
    output_predict[0] = df_log.iloc[0]
    upper_b = (df_log.shape[0] // timestamp) * timestamp
    # needs to be checked as there's more values that needs to be predicted other than the ne predicted by predict function
    output_predict[: df_log.shape[0], :], init_value = predict(modelnn, sess, df_log, True)

    output_predict[upper_b + 1 : df_log.shape[0] + 1], init_value = predict(
        modelnn, sess, df_log.iloc[upper_b:], True, init_value
    )
    df_log.loc[df_log.shape[0]] = output_predict[upper_b + 1 : df_log.shape[0] + 1][-1]
    date_ori.append(date_ori[-1] + timedelta(days=1))

    # In[10]:

    for i in range(future_day - 1):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict={
                modelnn.X: np.expand_dims(df_log.iloc[-timestamp:], axis=0),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[df_log.shape[0]] = out_logits[-1]
        df_log.loc[df_log.shape[0]] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(days=1))

    # In[11]:

    df_log = minmax.inverse_transform(output_predict)
    date_ori = pd.Series(date_ori).dt.strftime(date_format="%Y-%m-%d").tolist()

    # In[12]:

    def anchor(signal, weight):
        buffer = []
        last = signal[0]
        for i in signal:
            smoothed_val = last * weight + (1 - weight) * i
            buffer.append(smoothed_val)
            last = smoothed_val
        return buffer

    # In[13]:

    current_palette = sns.color_palette("Paired", 12)
    fig = plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    x_range_original = np.arange(data_pars.shape[0])
    x_range_future = np.arange(df_log.shape[0])
    ax.plot(x_range_original, data_pars.iloc[:, 1], label="true Open", color=current_palette[0])
    ax.plot(
        x_range_future, anchor(df_log[:, 0], 0.5), label="predict Open", color=current_palette[1]
    )
    ax.plot(x_range_original, data_pars.iloc[:, 2], label="true High", color=current_palette[2])
    ax.plot(
        x_range_future, anchor(df_log[:, 1], 0.5), label="predict High", color=current_palette[3]
    )
    ax.plot(x_range_original, data_pars.iloc[:, 3], label="true Low", color=current_palette[4])
    ax.plot(
        x_range_future, anchor(df_log[:, 2], 0.5), label="predict Low", color=current_palette[5]
    )
    ax.plot(x_range_original, data_pars.iloc[:, 4], label="true Close", color=current_palette[6])
    ax.plot(
        x_range_future, anchor(df_log[:, 3], 0.5), label="predict Close", color=current_palette[7]
    )
    ax.plot(x_range_original, data_pars.iloc[:, 5], label="true Adj Close", color=current_palette[8])
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

    # In[14]:

    fig = plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(x_range_original, data_pars.iloc[:, 1], label="true Open", color=current_palette[0])
    plt.plot(x_range_original, data_pars.iloc[:, 2], label="true High", color=current_palette[2])
    plt.plot(x_range_original, data_pars.iloc[:, 3], label="true Low", color=current_palette[4])
    plt.plot(x_range_original, data_pars.iloc[:, 4], label="true Close", color=current_palette[6])
    plt.plot(x_range_original, data_pars.iloc[:, 5], label="true Adj Close", color=current_palette[8])
    plt.xticks(x_range_original[::60], data_pars.iloc[:, 0].tolist()[::60])
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

    # In[15]:

    fig = plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    ax.plot(x_range_original, data_pars.iloc[:, -1], label="true Volume")
    ax.plot(x_range_future, anchor(df_log[:, -1], 0.5), label="predict Volume")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.xticks(x_range_future[::30], date_ori[::30])
    plt.title("overlap market volume")
    plt.show()

    # In[16]:

    fig = plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(x_range_original, data_pars.iloc[:, -1], label="true Volume")
    plt.xticks(x_range_original[::60], data_pars.iloc[:, 0].tolist()[::60])
    plt.legend()
    plt.title("true market volume")
    plt.subplot(1, 2, 2)
    plt.plot(x_range_future, anchor(df_log[:, -1], 0.5), label="predict Volume")
    plt.xticks(x_range_future[::60], date_ori[::60])
    plt.legend()
    plt.title("predict market volume")
    plt.show()

    # In[ ]:
    """
