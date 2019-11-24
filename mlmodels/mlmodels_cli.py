# -*- coding: utf-8 -*-
"""
Lightweight Functional interface to wrap access to Deep Learning, RLearning models.
Logic follows Scikit Learn API and simple for easy extentions logic.
Goal to facilitate Jupyter to Prod. models.


Models are stored in model_XX/  or in folder XXXXX
    module :  folder/mymodel.py, contains the methods, operations.
    model  :  Class in mymodel.py containing the model definition, compilation
   

models.py   #### Generic Interface
   module_load(model_uri)
   model_create(module)
   fit(model, module, session, data_pars, out_pars   )
   metrics(model, module, session, data_pars, out_pars)
   predict(model, module, session, data_pars, out_pars)
   save()
   load()


Example of custom model : model_tf/mymodels.py  : Allows to de-couple with Wrapper
  Class Model()  : Model definition
  fit(model, data_pars )               :  Fit wrapper
  predict(model, session, data_pars)   :
  stats(model)
  save(model, session)
  load(folder, load_pars)
  
  

####################################################################################################
######### Code sample  #############################################################################
from mlmodels.models import module_load, data_loader, create_model, fit, predict, stats

model_pars = { "learning_rate": 0.001, "num_layers": 1,
                  "size": ncol_input, "size_layer": 128, "output_size": ncol_output, "timestep": 4,  
                  "epoch": 2,}
data_pars = {}

module = models.module_load( model_uri="model_tf.1_lstm.py" )  #Load file definition
model =  models.model_create(module, model_pars)    # Create Model instance
sess =   models.fit(model, module, data_pars)       # fit the model
dict_stats = models.metrics( model, sess, ["loss"])     # get stats

model.save( "myfolder/", model, module, sess,)


model = module.load(folder)    #Create Model instance
module.predict(model, module, data_pars)     # predict pipeline



#df = data_loader(data_pars)

# module, model = module_load_full("model_tf.1_lstm.py", dict_pars= model_pars)  # Net



######### Command line sample  #####################################################################
####
python mlmodels/models.py  --do generate_config  --model_uri model_tf.1_lstm.py  --save_folder "c:\myconfig\" 


### RL model
python  models.py  --model_uri model_tf.rl.4_policygradient  --do test


### TF DNN model
python  models.py  --model_uri model_tf.1_lstm.py  --do test


## PyTorch models
python  models.py  --model_uri model_tch.mlp.py  --do test






"""
import sys
import argparse
import glob
import os
import re
import inspect
from argparse import Namespace
from importlib import import_module
import json


# from aapackage.mlmodel import util
# import pandas as pd
# import tensorflow as tf
# from os.path import join

####################################################################################################
from util import load_config, get_recursive_files, get_recursive_folder
from models import  module_load, module_load_full, model_create
from mlmodels import optim




def log(*args):
    print(  ",".join( [  x for x in args   ]  ))


 


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

    ### model params
    p.add_argument("--model_uri", default="model_tf.1_lstm.py",  help=".")
    p.add_argument("--dataname", default="dataset/google.csv",  help=".")

 
    ## data_pars
    p.add_argument("--data_path", default="dataset/GOOG-year_small.csv",  help="path of the training file")


    ## optim params
    p.add_argument("--ntrials", default=100, help='number of trials during the hyperparameters tuning')
    p.add_argument('--optim_engine', default='optuna',help='Optimization engine')
    p.add_argument('--optim_method', default='normal/prune',help='Optimization method')


    ## out params
    p.add_argument("--save_folder", default="ztest/",  help=".")
    p.add_argument("--load_folder", default="ztest/",  help=".")

    
    arg = p.parse_args()
    # arg = load_config(arg, arg.config_file, arg.config_mode, verbose=0)
    return arg

    
if __name__ == "__main__":
    arg = load_arguments()
    print(arg.do)

    if arg.do == "model_list"  :  #list all models in the repo
        folder = os_file_current_path()
        module_names = get_recursive_folder(folder, r"model*/*.py" )                       
        for t in module_names :
            print(t)
                    
                                 
    if arg.do == "testall"  :
        # test_all() # tot test all te modules inside model_tf
        test_all(folder=None)

                                 
    if arg.do == "test"  :
        test(arg.model_uri.replace(".py", "" ))  # '1_lstm'


    if arg.do == "fit"  :
        model_p, data_p, compute_p, out_p  = config_get_pars(arg.config_file, arg.config_mode)
        
        module = module_load(arg.modelname)  # '1_lstm.py
        model = model_create(model_p)   # Exact map JSON and paramters

        log("Fit")
        sess = module.fit(model, data_pars=data_p, compute_pars= compute_p)

        log("Save")
        save(f"{arg.save_folder}/{arg.modelname}", arg.modelname, sess )


    if arg.do == "predict"  :
        model_p, data_p, compute_p, out_p  = config_get_pars(arg.config_file, arg.config_mode)
        # module = module_load(arg.modelname)  # '1_lstm'
        module, model, session = load(arg.load_folder)
        module.predict(model, session, data_pars= data_p, compute_pars= compute_p, out_pars=out_p )


    if arg.do == "generate_config"  :
        print( arg.save_folder)
        config_generate_template(arg.model_uri, to_folder= arg.save_folder)


    if arg.do == "optim_test"  :
        optim.test_fast()


    if arg.do == "optim_test_all"  :
        optim.test_all()


    if arg.do == "optim_search"  :
        model_pars, data_pars, compute_pars = optim.config_get_pars(arg)
        print(model_pars, data_pars, compute_pars)

        res = optim(arg.modelname,
                    model_pars = model_pars,
                    ntrials = int(arg.ntrials),
                    compute_pars = compute_pars,
                    data_pars  = data_pars,
                    save_folder  = arg.save_folder,
                    log_folder   = arg.log_file)  # '1_lstm'

        print("#############  Finished OPTIMIZATION  ###############")
        print(res)










