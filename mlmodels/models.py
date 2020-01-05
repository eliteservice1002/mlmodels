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
#### generate config file
python mlmodels/models.py  --do generate_config  --model_uri model_tf.1_lstm.py  --save_folder "c:\myconfig\" 


#### Cusomt Directory Models
python mlmodels/models.py --do test  --model_uri "D:\_devs\Python01\gitdev\mlmodels\mlmodels\model_tf\1_lstm.py"


### RL model
python  models.py  --model_uri model_tf.rl.4_policygradient  --do test


### TF DNN model
python  models.py  --model_uri model_tf.1_lstm.py  --do test


## PyTorch models
python  models.py  --model_uri model_tch.mlp.py  --do test







"""
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)


import sys
import argparse
import glob
import os
import re
import inspect
from importlib import import_module
import json
from pathlib import Path

####################################################################################################
from mlmodels.util import load_config, get_recursive_files, get_recursive_folder
from mlmodels.util import load_tf, load_tch,  save_tf, save_tch,  os_package_root_path, log



####################################################################################################
#module = import_module("mlmodels.model_tf.1_lstm")
#print(module)
#ys.exit(0)






####################################################################################################
def module_load(model_uri="", verbose=0):
    """
      Load the file which contains the model description
      model_uri:  model_tf.1_lstm.py  or ABSOLUTE PATH
    """

    # print(os_file_current_path())
    model_uri = model_uri.replace("/", ".")
    module = None
    if verbose : print(model_uri)

    try :
      #### Import from package mlmodels sub-folder
      #module = import_module("mlmodels.model_tf.1_lstm")
      model_name = model_uri.replace(".py", "")

      module = import_module( f"mlmodels.{model_name}")
      
    except Exception as e1 :
      try :
        ### Add Folder to Path and Load absoluate path model
        path_parent = str( Path( model_uri).parent.absolute())
        sys.path.append( path_parent )
        # print(path_parent, sys.path)
        
        #### import model_tf.1_lstm
        model_name = Path(model_uri).stem  # remove .py
        model_name =   str(Path(model_uri).parts[-2]) +"."+ str(model_name )
        #print(model_name)
        module = import_module(model_name)
        
      except Exception as e2:
        raise NameError( f"Module {model_name} notfound, {e1}, {e2}")
        
    if verbose: print(module)
    return module



def module_load_full(model_uri="", model_pars=None, choice=None):
  """
    Create Instance of the model, module
    model_uri:  model_tf.1_lstm.py
  """
  module = module_load(model_uri=model_uri)
  model = module.Model(**model_pars)
  return module, model


def model_create(module, model_pars=None):
    """
      Create Instance of the model from loaded module
      model_pars : dict params
    """
    if model_pars is None :
      model_pars = module.get_pars()

    model = module.Model(**model_pars)
    return model



def fit(model, module, compute_pars=None, data_pars=None, out_pars=None,  **kwarg):
    """
    Wrap fit generic method
    :type model: object
    """
    return module.fit(model, data_pars, **kwarg)


def predict(model, module, sess=None, compute_pars=None, data_pars=None, out_pars=None,  **kwarg):
    """
       predict  using a pre-trained model and some data
    :param model:
    :param module:
    :param sess:
    :param data_pars:
    :param out_pars:
    :param kwarg:
    :return:
    """
    return module.predict(model, sess, data_pars, **kwarg)


def metrics(model, module, sess=None, compute_pars=None, data_pars=None, out_pars=None, **kwarg):
  val = module.metrics(model, sess, data_pars, **kwarg)
  return val


def load(folder_name, model_type="tf", filename=None, **kwarg):
    """
       Load model/session from files
    :param folder_name:
    :param model_type:
    :param filename:
    :param kwarg:
    :return:
    """
    if model_type == "tf" :
        return load_tf(folder_name)

    if model_type == "tch" :
        return load_tch(folder_name)

    if model_type == "pkl" :
        return load_pkl(folder_name)


def save(folder_name, modelname="model_default", model_type="tf",  model_session=None, ** kwarg):
    """
       Save model/session on disk
    :param folder_name:
    :param modelname:
    :param sess:
    :param kwarg:
    :return:
    """
    if model_type == "tf" :
      os.makedirs(folder_name, exist_ok = True)
      file_path = f"{folder_name}/{modelname}.ckpt"
      save_tf(model_session, file_path)
      print(file_path)
      return 1      


    if  "model_tch" in modelname :
        return 1


    if  "model_keras" in modelname :
        return 1


    if  "model_" in modelname :
        return 1





####################################################################################################
####################################################################################################
def test_all(folder=None):
    if folder is None :
       folder =  os_package_root_path() +  "/model_tf/"
            
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
          module = module_load(model_uri= module_name)
          # module = import_module(f'{folder}.{module_name.replace(".py", "")}')
          module.test()
          del module
        except Exception as e:
          print("Failed", e)



def test(modelname):
    print(model_uri)
    try :
      module = module_load( modelname , verbose=1)
      print(module)
      module.test()
      del module
    except Exception as e:
      print("Failed", e)




####################################################################################################
############CLI Command ############################################################################
def config_get_pars(config_file, config_mode) :
   """ 
     load JSON and output the params
   """
   js = json.load(open(config_file, 'r'))  #Config
   js = js[config_mode]  #test /uat /prod
   model_p = js.get("model_pars")
   data_p = js.get("data_pars")
   compute_p  = js.get("compute_pars")
   out_p = js.get("out_pars")

   return model_p, data_p, compute_p, out_p
                                 
   
def config_generate_template(modelname, to_folder="ztest/") :
  """
    Generate config file from code source
    config_generate_template("model_tf.1_lstm", to_folder="ztest/")

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
  model_pars = {"model_pars" :     args,
                  "data_pars" :      {},
                  "compute_pars":    {},
                  "out_pars" :       {}
               }

  modelname = modelname.replace(".py", "").replace(".", "-" )
  fname = os.path.join( to_folder , f"{modelname}_config.json" )
  os.makedirs(to_folder, exist_ok=True)
  json.dump(model_pars, open( fname, mode="w"))
  print(fname)



def cli_load_arguments(config_file= None):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    if config_file is None  :
      cur_path = os.path.dirname(os.path.realpath(__file__))
      config_file = os.path.join(cur_path, "models_config.json")
    # print(config_file)

    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default=config_file, help="Params File")
    p.add_argument("--config_mode", default="test", help="test/ prod /uat")
    p.add_argument("--log_file", help="log.log")
    p.add_argument("--do", default="test", help="test")

    ##### model pars
    p.add_argument("--model_uri", default="model_tf/1_lstm.py",  help=".")
    p.add_argument("--load_folder", default="ztest/",  help=".")

    ##### data pars
    p.add_argument("--dataname", default="dataset/google.csv",  help=".")


    ##### compute pars


    ##### out pars
    p.add_argument("--save_folder", default="ztest/",  help=".")
    
    arg = p.parse_args()
    # arg = load_config(arg, arg.config_file, arg.config_mode, verbose=0)
    return arg


                                 

  
def main():
    arg = cli_load_arguments()
    print(arg.do)

    if arg.do == "model_list"  :  #list all models in the repo
        folder = os_package_root_path()
        module_names = get_recursive_folder(folder, r"model*/*.py" )                       
        for t in module_names :
            print(t)
                    
                                 
    if arg.do == "testall"  :
        # test_all() # tot test all te modules inside model_tf
        test_all(folder=None)

                                 
    if arg.do == "test"  :
        test(arg.model_uri)  # '1_lstm'


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

if __name__ == "__main__":
    main()
