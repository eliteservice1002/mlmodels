# -*- coding: utf-8 -*-
"""
Lightweight Functional interface to wrap Hyper-parameter Optimization


###### Model param search test
python optim.py --do test


##### #for normal optimization search method
python optim.py --do search --ntrials 1  --config_file optim_config.json --optim_method normal


###### for pruning method
python optim.py --do search --ntrials 1  --config_file optim_config.json --optim_method prune



###### HyperParam standalone run
python optim.py --modelname model_tf.1_lstm.py  --do test

python optim.py --modelname model_tf.1_lstm.py  --do search



### Distributed
https://optuna.readthedocs.io/en/latest/tutorial/distributed.html
{ 'distributed' : 1,
  'study_name' : 'ok' , 
  'storage' : 'sqlite'
}                                       
                                       


###### 1st engine is optuna
https://optuna.readthedocs.io/en/stable/installation.html
https://github.com/pfnet/optuna/blob/master/examples/tensorflow_estimator_simple.py
https://github.com/pfnet/optuna/tree/master/examples



"""
import argparse
import os
import re
import json


import pandas as pd
import optuna
####################################################################################################
# from mlmodels import models
from mlmodels.models import model_create, module_load, save
from mlmodels.util import os_package_root_path, log
#print(os_package_root_path())

####################################################################################################

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False





####################################################################################################
def optim(modelname="model_tf.1_lstm.py",
          model_pars= {},
          data_pars = {},
          compute_pars={"method": "normal/prune"},
          save_path="/mymodel/", log_path="", ntrials=2) :
    """
    Generic optimizer for hyperparamters
    Parameters
    ----------
    modelname : The default is "model_tf.1_lstm.py".
    model_pars : TYPE, optional
    data_pars : TYPE, optional
    compute_pars : TYPE, optional
    DESCRIPTION. The default is {"method": "normal/prune"}.
    save_path : TYPE, optional The default is "/mymodel/".
    log_path : TYPE, optional. The default is "".
    ntrials : TYPE, optional. The default is 2.
    Returns : None

    """
    log("model_pars", model_pars)
    if compute_pars["engine"] == "optuna" :
        return optim_optuna(modelname,  model_pars, data_pars, compute_pars,
                            save_path, log_path, ntrials)
    return None




def optim_optuna(modelname="model_tf.1_lstm.py",
                 model_pars= {},
                 data_pars = {},
                 compute_pars={"method" : "normal/prune"},
                 save_path="/mymodel/", log_path="", ntrials=2) :
    """
       Interface layer to Optuna  for hyperparameter optimization
       return Best Parameters

    optuna create-study --study-name "distributed-example" --storage "sqlite:///example.db"

    https://optuna.readthedocs.io/en/latest/tutorial/distributed.html
     if __name__ == '__main__':
    study = optuna.load_study(study_name='distributed-example', storage='sqlite:///example.db')
    study.optimize(objective, n_trials=100)


    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam']) # Categorical parameter
    num_layers = trial.suggest_int('num_layers', 1, 3)      # Int parameter
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)      # Uniform parameter
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)      # Loguniform parameter
    drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1) # Discrete-uniform parameter
    

    """
    module = module_load(modelname)
    log(module)
    
    def objective(trial):
        print("check", module)
        param_dict =  module.get_pars(choice="test",)
        # print([param_dict])

        for t,p  in model_pars.items():
            #p = model_pars[t]
            x = p['type']
  
            if   x=='log_uniform':       pres = trial.suggest_loguniform(t,p['range'][0], p['range'][1])
            elif x=='int':               pres = trial.suggest_int(t,p['range'][0], p['range'][1])
            elif x=='categorical':       pres = trial.suggest_categorical(t,p['value'])
            elif x=='discrete_uniform':  pres = trial.suggest_discrete_uniform(t, p['init'],p['range'][0],p['range'][1])
            elif x=='uniform':           pres = trial.suggest_uniform(t,p['range'][0], p['range'][1])
            else:
                raise Exception( f'Not supported type {x}')
                pres = None

            param_dict[t] = pres

        model = model_create(module, param_dict)   # module.Model(**param_dict)
        print(model)
        # df = data_loader(data_pars)
        
        sess = module.fit(model, data_pars=data_pars, compute_pars= compute_pars)
        #return 1
        metrics = module.metrics(model, sess, data_pars=data_pars)  #Dictionnary
        # stats = model.stats["loss"]
        del sess
        del model
        try :
           module.reset_model()  # Reset Graph for TF
        except Exception as e :
           print(e)

        return metrics["loss"]

    log("###### Hyper-optimization through study   ####################################")
    pruner = optuna.pruners.MedianPruner() if compute_pars["method"] =='prune' else None
          
    if compute_pars.get("distributed") is not None :
          # study = optuna.load_study(study_name='distributed-example', storage='sqlite:///example.db')
          try :
             study = optuna.load_study(study_name= compute_pars['study_name'] ,
                                       storage=compute_pars['storage'] )
          except:
             study = optuna.create_study(pruner=pruner, study_name= compute_pars['study_name'] ,
                                         storage=compute_pars['storage'] )
    else :           
         study = optuna.create_study(pruner=pruner)


    study.optimize(objective, n_trials=ntrials)  # Invoke optimization of the objective function.
    log("Optim, finished", n=35)
    param_dict_best =  study.best_params
    # param_dict.update(module.config_get_pars(choice="test", )
    ###############################################################################


    log("### Run Model with best   ################################################")
    model = model_create( module, model_pars=param_dict_best)
    sess = module.fit(model,  data_pars=data_pars, compute_pars=compute_pars)

    log("#### Saving     ###########################################################")
    modelname = modelname.replace(".", "-") # this is the module name which contains .
    save( save_path, modelname, sess, model=model )


    log("### Save Stats   ##########################################################")
    study_trials = study.trials_dataframe()
    study_trials.to_csv(f"{save_path}/{modelname}_study.csv")

    param_dict_best["best_value"] = study.best_value
    # param_dict["file_path"] = file_path
    json.dump( param_dict_best, open(f"{save_path}/{modelname}_best-params.json", mode="w") )

    return param_dict_best






####################################################################################################

def test_all():
    pars =  {
        "learning_rate": {"type": "log_uniform", "init": 0.01,  "range" : [0.001, 0.1] },
        "num_layers":    {"type": "int", "init": 2,  "range" :[2, 4] },
        "size":    {"type": "int", "init": 6,  "range" :[6, 6] },
        "output_size":    {"type": "int", "init": 6,  "range" : [6, 6] },

        "size_layer":    {"type" : "categorical", "value": [128, 256 ] },
        "timestep":      {"type" : "categorical", "value": [5] },
        "epoch":         {"type" : "categorical", "value": [2] }
    }

    data_path = os_package_root_path('dataset/GOOG-year_small.csv')

    res = optim('model_tf.1_lstm', model_pars=pars,
                data_pars={"data_path": data_path, "data_type": "pandas"},
                ntrials=2,
                save_path="ztest/optuna_1lstm/",
                log_path="ztest/optuna_1lstm/",
                compute_pars={"engine": "optuna" ,  "method" : "prune"} )

    return res


def test_fast(ntrials=2):

    modelname = 'model_tf.1_lstm'

    model_pars = {
        "learning_rate": {"type": "log_uniform", "init": 0.01,  "range" : [0.001, 0.1] },
        "num_layers":    {"type": "int", "init": 2,  "range" :[2, 4] },
        "size":    {"type": "int", "init": 6,  "range" :[6, 6] },
        "output_size":    {"type": "int", "init": 6,  "range" : [6, 6] },

        "size_layer":    {"type" : "categorical", "value": [128, 256 ] },
        "timestep":      {"type" : "categorical", "value": [5] },
        "epoch":         {"type" : "categorical", "value": [2] }
    }
    log( "model details" , modelname, model_pars ) 


    data_path = os_package_root_path('dataset/GOOG-year_small.csv')
    log( "data_path" , data_path )


    path_curr = os.getcwd()
    path_save = f"{path_curr}/ztest/optuna_1lstm/" 
    os.makedirs(path_save, exist_ok=True)
    log("path_save", path_save)



    res = optim(modelname    = modelname,
                model_pars   = model_pars,
                data_pars    = {"data_path": data_path, "data_type": "pandas"},
                ntrials      = ntrials,
                save_path    = path_save,
                log_path     = path_save,
                compute_pars = {"engine": "optuna" ,  "method" : "prune"} )

    log("Finished OPTIMIZATION",n =30)
    print(res)




####################################################################################################
####################################################################################################
def cli_load_arguments(config_file= None):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    if config_file is None  :
      cur_path = os.path.dirname(os.path.realpath(__file__))
      config_file = os.path.join(cur_path, "optim_config.json")
    # print(config_file)

    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default=config_file, help="Params File")
    p.add_argument("--config_mode", default="test", help="test/ prod /uat")
    p.add_argument("--log_file", help="File to save the logging")

    p.add_argument("--do", default="test", help="what to do test or search")


    ###### model_pars
    p.add_argument("--modelname", default="model_tf.1_lstm.py",  help="name of the model to be tuned this name will be used to save the model")


    ###### data_pars
    p.add_argument("--data_path", default="dataset/GOOG-year_small.csv",  help="path of the training file")


    ###### compute params
    p.add_argument("--ntrials", default=100, help='number of trials during the hyperparameters tuning')
    p.add_argument('--optim_engine', default='optuna',help='Optimization engine')
    p.add_argument('--optim_method', default='normal/prune',help='Optimization method')


    ###### out params
    p.add_argument('--save_path', default='ztest/search_save/',help='folder that will contain saved version of best model')


    args = p.parse_args()
    # args = load_config(args, args.config_file, args.config_mode, verbose=0)
    return args



def config_get_pars(arg) :

   js = json.load(open(arg.config_file, 'r'))  #Config     
   js = js[arg.config_mode]  #test /uat /prod
   model_pars = js.get("model_pars")
   data_pars = js.get("data_pars")
   compute_pars = js.get("compute_pars")

   return model_pars, data_pars, compute_pars



####################################################################################################
####################################################################################################
def main():
    arg = cli_load_arguments()
    
    # import logging
    # logging.getLogger("tensorflow").setLevel(logging.ERROR)

    if arg.do == "test"  :
        test_fast()


    if arg.do == "test_all"  :
        test_all()


    if arg.do == "search"  :
        model_pars, data_pars, compute_pars = config_get_pars(arg)
        log(model_pars, data_pars, compute_pars)
        log("############# OPTIMIZATION Start  ###############")
        res = optim(arg.modelname,
                    model_pars = model_pars,
                    ntrials = int(arg.ntrials),
                    compute_pars = compute_pars,
                    data_pars  = data_pars,
                    save_path  = arg.save_path,
                    log_path   = arg.log_file)  # '1_lstm'

        log("#############  OPTIMIZATION End ###############")
        log(res)


if __name__ == "__main__":
    main()










####################################################################################################
####################################################################################################


"""
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

def data_loader(data_pars):
    if data_pars["data_type"] == "pandas" :
      df = pd.read_csv(data_pars["data_path"])

    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

    minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype('float32'))
    df_log = minmax.transform(df.iloc[:, 1:].astype('float32'))
    df_log = pd.DataFrame(df_log)
    return df_log
"""




