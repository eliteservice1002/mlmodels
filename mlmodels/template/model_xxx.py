# coding: utf-8
"""
Generic template for new model.
Check parameters template in models_config.json


"model_pars":   { "learning_rate": 0.001, "num_layers": 1, "size": 6, "size_layer": 128, "output_size": 6, "timestep": 4, "epoch": 2 },
"data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] },
"compute_pars": { "distributed": "mpi", "epoch": 10 },
"out_pars":     { "out_path": "dataset/", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] }



"""
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

import os, sys, inspect
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# import tensorflow as tf





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
     get the module package root folder
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


class to_namespace(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

  def get(self, key):
    return self.__dict__.get(key)



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
    
    self.X = 0
    self.Y = 0
    
    ### Model Structure        ################################
    """

    """
    
    ### Loss    ##############################################
    self.cost =   0
    self.optimizer = 0




def fit(model, data_pars={}, out_pars={}, compute_pars={}, **kwargs):
  """

  :param model:    Class model
  :param data_pars:  dict of
  :param out_pars:
  :param compute_pars:
  :param kwargs:
  :return:
  """
  # df = get_dataset(data_pars)
  #print(df.head(5))
  #msample = df.shape[0]

  nlog_freq = compute_pars.get("nlog_freq", 100)
  epoch     = compute_pars.get("epoch", 1)
  msample   = data_pars.get("msample", 1)

  ######################################################################
  sess =None
  for i in range(epoch):
    total_loss = 0.0
    ######## Model specific  ########################################




    ####### End Model specific    ##################################
    model.stats["loss"] = total_loss
    
    if (i + 1) % nlog_freq == 0:
      print("epoch:", i + 1, "avg loss:", total_loss)
  return sess



def metrics(model, sess=None, data_pars=None, out_pars=None):
  """
       Return metrics of the model stored
    #### SK-Learn metrics
    # Compute stats on training
    """
  return model.stats


def predict(model, sess, data_pars=None, out_pars=None, compute_pars=None,
            get_hidden_state=False, init_value=None):

  """
     Preidction results
  """
  #############################################################
  df = get_dataset(data_pars)
  print(df, flush=True)
  
  #############################################################


  #############################################################
  return predict


def reset_model():
  pass


####################################################################################################
def get_dataset(data_pars=None):
  """
    JSON data_pars to get dataset
    "data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas",
    "size": [0, 0, 6], "output_size": [0, 6] },
  """
  d = to_namespace(data_pars)
  print(d)
  df = None

  if d.data_type == "pandas" :
    df = pd.DataFrame(d.data_path)
  
  #######################################
  return df


def get_pars(choice="test", **kwargs):
  # Get sample parameters of the model
  if choice == "test":
    p = {"learning_rate": 0.001, "num_layers": 1, "size": None, "size_layer": 128,
         "output_size": None, "timestep": 4, "epoch": 2,}
    
    ### Overwrite by manual input
    for k, x in kwargs.items():
      p[k] = x
    
    return p



###############################################################################################
def test_local(data_path="dataset/GOOG-year.csv"):
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


def test_generic(data_path="dataset/GOOG-year.csv", out_path="", reset=True):
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





if __name__ == "__main__":
  test_local()
  
  
  
  
  
  
  
  