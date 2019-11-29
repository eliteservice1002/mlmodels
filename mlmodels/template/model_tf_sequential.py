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
def os_module_path():
  current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  parent_dir = os.path.dirname(current_dir)
  # sys.path.insert(0, parent_dir)
  return parent_dir


def os_file_path(data_path):
  from pathlib import Path
  data_path = os.path.join(Path(__file__).parent.parent.absolute(), data_path)
  print(data_path)
  return data_path


def os_package_root_path(filepath, sublevel=0, path_add=""):
  """
    :param filepath:
    :param sublevel:  level 0 : current path, level 1 : 1 level above
    :param path_add:
    :return:
    """
  from pathlib import Path
  path = Path(filepath).parent
  for i in range(1, sublevel + 1):
    path = path.parent
  
  path = os.path.join(path.absolute(), path_add)
  return path


# print("check", os_package_root_path(__file__, sublevel=1) )


def log(*s, n=0, m=1):
  sspace = "#" * n
  sjump = "\n" * m
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
    self.stats = {"loss": 0.0,
                  "loss_history": []}
    
    self.X = tf.placeholder(tf.float32, (None, None, size))
    self.Y = tf.placeholder(tf.float32, (None, output_size))
    
    
    ### Model Structure        ################################
    """
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
    """
    
    ### Loss    ##############################################
    self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
    self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self.cost)




def fit(model, data_pars, out_pars=None, compute_pars={}, **kwargs):
  df = get_dataset(data_pars)
  print(df.head(5))
  msample = df.shape[0]
  nlog_freq = compute_pars.get("nlog_freq", 100)
  
  ######################################################################
  sess = tf.compat.v1.InteractiveSession()
  sess.run(tf.compat.v1.global_variables_initializer())
  for i in range(model.epoch):
    total_loss = 0.0
    
    ######## Model specific  ########################################
    """
    init_value = np.zeros((1, model.hidden_layer_size))
    for k in range(0, msample - 1, model.timestep):
      index = min(k + model.timestep, msample - 1)
      batch_x = np.expand_dims(df.iloc[k:index, :].values, axis=0)
      batch_y = df.iloc[k + 1: index + 1, :].values
      last_state, _, loss = sess.run(
        [model.last_state, model.optimizer, model.cost],
        feed_dict={model.X: batch_x, model.Y: batch_y, model.hidden_layer: init_value},
      )
      init_value = last_state
      total_loss += loss
    """
    ####### End Model specific    ##################################
    
    
    total_loss /= msample // model.timestep
    model.stats["loss"] = total_loss
    
    if (i + 1) % nlog_freq == 0:
      print("epoch:", i + 1, "avg loss:", total_loss)
  return sess


def metrics(model, sess=None, data_pars=None, out_pars=None):
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


def predict(model, sess, data_pars=None, out_pars=None, compute_pars=None,
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
    output_predict[1:  model.timestep + 1] = out_logits
  
  else:
    for k in range(0, (df.shape[0] // model.timestep) * model.timestep, model.timestep):
      out_logits, last_state = sess.run(
        [model.logits, model.last_state],
        feed_dict={model.X: np.expand_dims(df.iloc[k: k + model.timestep].values, axis=0),
                   model.hidden_layer: init_value,
                   },
      )
      init_value = last_state
      output_predict[k + 1: k + model.timestep + 1] = out_logits
  
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
  print(filename)
  print(df.head(5))
  
  minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype("float32"))
  df_log = minmax.transform(df.iloc[:, 1:].astype("float32"))
  df_log = pd.DataFrame(df_log)
  return df_log


def get_pars(choice="test", **kwargs):
  # output parms
  # print(kwargs)
  if choice == "test":
    p = {"learning_rate": 0.001,
         "num_layers": 1,
         "size": None,
         "size_layer": 128,
         "output_size": None,
         "timestep": 4,
         "epoch": 2,
         }
    
    ### Overwrite by manual input
    for k, x in kwargs.items():
      p[k] = x
    
    return p


####################################################################################################
def test(data_path="dataset/GOOG-year.csv", out_path="", reset=True):
  """
       Using mlmodels package method
       path : mlmodels/mlmodels/dataset/
       from ../../model_tf

    """
  data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
  print(data_path)
  
  log("############# Data, Params preparation   #################")
  data_pars = {"data_path": data_path, "data_type": "pandas"}
  out_pars = {"path": data_path + out_path}
  compute_pars = {}
  
  df = get_dataset(data_pars)
  model_pars = get_pars("test", size=df.shape[1], output_size=df.shape[1])
  
  log("############ Model preparation   #########################")
  from mlmodels.models import module_load_full, fit, predict
  module, model = module_load_full("model_tf.1_lstm", model_pars)
  print(module, model)
  
  log("############ Model fit   ##################################")
  sess = fit(model, module, data_pars=data_pars, out_pars=out_pars, compute_pars={})
  print("fit success", sess)
  
  log("############ Prediction##########################")
  preds = predict(model, module, sess, data_pars=data_pars,
                  out_pars=out_pars, compute_pars=compute_pars)
  print(preds)


def test2(data_path="dataset/GOOG-year.csv"):
  """
      Using this file methods
    """
  #### path to local package roots
  data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
  print(data_path)
  
  data_pars = {"data_path": data_path, "data_type": "pandas"}
  out_pars = {"path": data_path}
  compute_pars = {}
  
  ###Need to get variable size to initiatlize the model
  df = get_dataset(data_pars)
  model_pars = get_pars("test", size=df.shape[1], output_size=df.shape[1])
  
  #### Model setup, fit, predict
  model = Model(**model_pars)
  sess = fit(model, data_path=data_pars)
  predictions = predict(model, sess, data_pars)
  print(predictions)


if __name__ == "__main__":
  test2()
  
  
  
  
  
  
  
  