# -*- coding: utf-8 -*-
"""
Lightweight Functional interface to wrap access to Deep Learning, RLearning models.
Logic follows Scikit Learn API and simple for easy extentions.Logic

Models are stored in model/
folder/mymodel.py

   Class Model( )
   def fit(model, )
   def predict(model, sess, )
   def test()

######### CLI sample  ##########################################################
python models.py  --do test

python models.py  --do generate_config  --modelname model_tf.1_lstm.py  --save_folder "c:\myconfig\"






######### Code sample  ##########################################################
from models import create
# module, model = create_full("model_tf.1_lstm.py", dict_params= model_params)  # Net

model_params =  { "learning_rate": 0.001, "num_layers": 1,
            "size": ncol_input, "size_layer": 128, "output_size": ncol_output, "timestep": 4,  
            "epoch": 2,}
data_params = {}

module = module_load(modelname="model_tf.1_lstm.py")
model = module.create(module, model_params)

df = data_loader(data_params)
sess = module.fit(model, df)
stats = model.stats["loss"]
module.save("myfolder/")



######### Command line sample  #################################################
### RL model
python  models.py  --modelname model_tf.rl.4_policygradient  --do test


### TF DNN model
python  models.py  --modelname model_tf.1_lstm.py  --do test


## PyTorch models
python  models.py  --modelname model_tch.mlp.py  --do test



"""
import argparse
import glob
import os
import re
from importlib import import_module
import json



# from aapackage.mlmodel import util
# import pandas as pd
# import tensorflow as tf



#################################################################################
from util import load_config, get_recursive_files, get_recursive_folder

def log(*args):
    print(  ",".join( [  x for x in args   ]  ))


#################################################################################
def module_load(modelname=""):
    """
      Load the file which contains the model description
      modelname:  model_tf.1_lstm.py
    """
    print(modelname)
    modelname = modelname.replace(".py", "")

    try :
      module = import_module(modelname)
    except Exception as e :
      raise NameError("Module {} notfound, {}".format(modelname, e))
    return module


def create(module, model_params=None):
    """
      Create Instance of the model from loaded module
      model_params : dict params
    """
    if model_params is None :
      model_params = module.get_params()

    model = module.Model(**model_params)
    return model


def create_full(modelname="", model_params=None, choice=None):
    """
      Create Instance of the model, module
      modelname:  model_tf.1_lstm.py
    """
    module = module_load(modelname=modelname)
    model = module.Model(**model_params)
    return module, model


def fit(model, module, data_params, **kwarg):
    return module.fit(model, data_params, **kwarg)


def predict(model, module, sess=None, data_params=None, **kwarg):
    return module.predict(model, sess, data_params, **kwarg)


def load(folder_name, model_type="tf", filename=None, **kwarg):
    if model_type == "tf" :
        return load_tf(folder_name)

    if model_type == "tch" :
        return load_tch(folder_name)

    if model_type == "pkl" :
        return load_pkl(folder_name)


def save(folder_name, modelname=None,  sess=None, ** kwarg):
    if "model_tf" in modelname :
      #save_folder = save_folder + "/" + modelname
      if not(os.path.isdir(folder_name)):
        os.makedirs(folder_name)
      file_path = f"{folder_name}/{modelname}.ckpt"
      save_tf(sess, file_path)
      print(file_path)
      return 1      


    if  "model_tch" in modelname :
        return 1


    if  "model_keras" in modelname :
        return 1


    if  "model_" in modelname :
        return 1



########## TF specific #############################################################################
def load_tf(foldername, filename):
    """
    https://www.mlflow.org/docs/latest/python_api/mlflow.tensorflow.html#
    
    
   """
    import mlflow.tensorflow
    import tensorflow as tf
    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)
    with tf_graph.as_default():
        signature_def = mlflow.tensorflow.load_model(model_uri="model_uri",
                               tf_sess=tf_sess)
        input_tensors = [tf_graph.get_tensor_by_name(input_signature.name)
                        for _, input_signature in signature_def.inputs.items()]
        output_tensors = [tf_graph.get_tensor_by_name(output_signature.name)
                          for _, output_signature in signature_def.outputs.items()]
    return input_tensors, output_tensors



def save_tf(sess, file_path):
    import tensorflow as tf
    saver = tf.train.Saver()
    return saver.save(sess, file_path)




########## pyTorch specific ########################################################################
def load_tch(foldername, filename):
    return 1


def save_tch(foldername, filename):
    return 1




########## Other model specific ####################################################################
def load_pkl(folder_name, filename=None) :
    pass






####################################################################################################
####################################################################################################
def test_all(folder=None):
    if folder is None :
       folder =  folder_file() +  "/model_tf/"
            
    module_names = get_recursive_files(folder, r"[0-9]+_.+\.py$")
    module_names.sort()
    print(module_names)

    failed_scripts = []

    if folder == "model_tf" :
       import tensorflow as tf

    for module_name in module_names:
        print("#######################")
        print(module_name)
        try :
          module = import_module(f'{folder}.{module_name.replace(".py", "")}')
          module.test()
          del module
        except Exception as e:
          print("Failed", e)



def test(modelname):
        try :
          module = import_module( modelname )
          module.test()
          del module
        except Exception as e:
          print("Failed", e)




####################################################################################################
############CLI Command ############################################################################
def load_arguments(config_file= None ):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    if config_file is None  :
      cur_path = os.path.dirname(os.path.realpath(__file__))
      config_file = os.path.join(cur_path, "models_config.json")
    print(config_file)

    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default=config_file, help="Params File")
    p.add_argument("--config_mode", default="test", help="test/ prod /uat")
    p.add_argument("--log_file", help="log.log")

    p.add_argument("--do", default="test", help="test")
    p.add_argument("--modelname", default="model_tf.1_lstm.py",  help=".")
    p.add_argument("--dataname", default="dataset/google.csv",  help=".")
                                 
    p.add_argument("--save_folder", default="ztest/",  help=".")
    p.add_argument("--load_folder", default="ztest/",  help=".")
                                 
    arg = p.parse_args()
    # arg = load_config(arg, arg.config_file, arg.config_mode, verbose=0)
    return arg


                                 
def get_params(arg) :
   """  From CLI Input to JSON format
      JSON should map EXACTLY the model input
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))

   """
   js = json.load(open(arg.config_file, 'r'))  #Config     
   js = js[arg.config_mode]  #test /uat /prod                              
   model_p = js.get("model_params")
   data_p = js.get("data_params")

   return model_p, data_p
                                 


def folder_file() :
  # Path of current file
  from pathlib import Path
  return Path().absolute()  

                                 
                                 
                                 
def generate_config_file(modelname, to_folder="ztest/") :
  """
    Generate config file from code source

    generate_config_file("model_tf.1_lstm", to_folder="ztest/")

  """
  import inspect                               
  module = module_load(modelname)
  signature = inspect.signature( module.Model )
  args = {
        k: v.default if v.default is not inspect.Parameter.empty else None
        for k, v in signature.parameters.items()
        # if v.default is not inspect.Parameter.empty
  }
                                 
  # args = inspect.getargspec(module.Model)
  model_params = {"model_params" : args,
                  "data_params" : {},
                  "optim_params": {}                 
                 }

  modelname.replace(".", "")
  fname = f"{to_folder}/{modelname}_config.json"
  json.dump(model_params, open( fname, mode="w"))
  print(fname)




                                 
                                 
if __name__ == "__main__":
    # test_all() # tot test all te modules inside model_tf
    arg = load_arguments()
    print(arg.do)

    if arg.do == "model_list"  :  #list all models
        folder = folder_file()                         
        module_names = get_recursive_folder(folder, r"model*/*.py" )                       
        for t in module_names :
            print(t)
                    
                                 
    if arg.do == "testall"  :    
        test_all(folder=None)

                                 
    if arg.do == "test"  :
        test(arg.modelname.replace(".py", "" ))  # '1_lstm'


    if arg.do == "fit"  :
        model_params, data_params = get_params(arg)
        module = module_load(arg.modelname)  # '1_lstm.py

        model = module.Model(**model_params)   #Exact map JSON and paramters

        log("Fit")
        sess = module.fit(model, data_params)

        log("Save")
        save(f"{arg.save_folder}/{arg.modelname}", arg.modelname, sess )


                                 
    if arg.do == "predict"  :
        model_params, data_params = get_params(arg)      
                                 
        module = module_load(arg.modelname)  # '1_lstm'
        model = load(arg.load_folder)
        module.predict(model, data_params, data_params )



    if arg.do == "generate_config"  :
        generate_config_file(arg.modelname, to_folder= arg.save_folder) 













