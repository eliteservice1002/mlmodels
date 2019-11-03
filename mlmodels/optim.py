# -*- coding: utf-8 -*-
"""
Lightweight Functional interface to wrap Hyper-parameter Optimization

1st engine is optuna
https://optuna.readthedocs.io/en/stable/installation.html
https://github.com/pfnet/optuna/blob/master/examples/tensorflow_estimator_simple.py
https://github.com/pfnet/optuna/tree/master/examples


###### Model param search

#for normal optimization search method
python optim.py --do search --ntrials 1  --config_file data.json --optim_method normal

# for pruning method
python optim.py --do search --ntrials 1  --config_file data.json --optim_method prune


###### HyperParam standalone run
python optim.py --modelname model_tf.1_lstm.py  --do test

python optim.py --modelname model_tf.1_lstm.py  --do search



"""
import argparse
import glob
import os
import re
from importlib import import_module
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd


import optuna
####################################################################################################


from util import load_config, to_namespace
from models import create, module_load, save
####################################################################################################


def optim(modelname="model_tf.1_lstm.py",
                 model_params= {},
                 data_params = {},
                 optim_params={"method": "normal/prune"},
                 save_folder="/mymodel/", log_folder="", ntrials=2) :
    """
    Generic interface of  Hyper-optimize
    :param modelname:
    :param pars:
    :param df:
    :param optim_engine:
    :param optim_method:
    :param save_folder:
    :param log_folder:
    :param ntrials:
    :return:
    """
    print(model_params)
    if optim_params["engine"] == "optuna" :
        return optim_optuna(modelname,  model_params, data_params, optim_params,
                            save_folder, log_folder, ntrials)
    return None


def optim_optuna(modelname="model_tf.1_lstm.py",
                 model_params= {},
                 data_params = {},
                 optim_params={"method" : "normal/prune"},
                 save_folder="/mymodel/", log_folder="", ntrials=2) :
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

    def objective(trial):
        param_dict =  module.get_params(choice="test", ncol_input=data_params.shape[1],
                                        ncol_output=data_params.shape[1])
        for t,p  in model_params.items():
            pres = None
            #p = model_params[t]
            x = p['type']

            if x=='log_uniform':
                pres = trial.suggest_loguniform(t,p['range'][0], p['range'][1])

            elif x=='int':
                pres = trial.suggest_int(t,p['range'][0], p['range'][1])

            elif x=='categorical':
                pres = trial.suggest_categorical(t,p['value'])

            elif x=='discrete_uniform':
                pres = trial.suggest_discrete_uniform(t, p['init'],p['range'][0],p['range'][1])

            elif x=='uniform':
                pres = trial.suggest_uniform(t,p['range'][0], p['range'][1])

            else:
                raise Exception('Not supported type {}'.format(p['type']))

            param_dict[t] = pres

        model = module.Model(**param_dict)

        df = data_loader(data_params)
        sess = module.fit(model, df)
        stats = model.stats["loss"]
        del sess
        del model
        try :
           module.reset_model()
           # tf.reset_default_graph()
        except Exception as e :
           print(e)

        return stats

    ###### Hyper-optimization through study   ####################################
    if optim_params["method"] =='prune':
        study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study()  # Create a new study.

    study.optimize(objective, n_trials=ntrials)  # Invoke optimization of the objective function.
    param_dict =  study.best_params
    param_dict.update(module.get_params(choice="test", ncol_input=data_params.shape[1],
                                        ncol_output=data_params.shape[1]))


    ### Run Model with best   ###################################################
    model = module.Model(**param_dict)
    df = data_loader(data_params)
    sess = module.fit(model,  df)


    #### Saving     #############################################################
    modelname = modelname.replace(".", "-") # this is the module name which contains .
    module.save( save_folder, modelname, sess, model )


    ### Save Stats   ############################################################
    study_trials = study.trials_dataframe()
    study_trials.to_csv(f"{save_folder}/{modelname}_study.csv")

    param_dict["best_value"] = study.best_value
    param_dict["file_path"] = file_path
    json.dump( param_dict,  f"{save_folder}/{modelname}_params.json" )

    return param_dict



##########################################################################################################
def load_arguments(config_file= None ):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    if config_file is None  :
      cur_path = os.path.dirname(os.path.realpath(__file__))
      config_file = os.path.join(cur_path, "optim_config_default.json")
    print(config_file)

    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default=config_file, help="Params File")
    p.add_argument("--config_mode", default="test", help="test/ prod /uat")
    p.add_argument("--log_file", help="File to save the logging")

    p.add_argument("--do", default="test", help="what to do test or search")

    ## optim params
    p.add_argument("--ntrials", default=100, help='number of trials during the hyperparameters tuning')
    p.add_argument('--optim_engine', default='optuna',help='Optimization engine')
    p.add_argument('--optim_method', default='normal/prune',help='Optimization method')
    p.add_argument('--save_folder', default='model_save',help='folder that will contain saved version of best model')


    ## model_params
    p.add_argument("--modelname", default="model_tf.1_lstm.py",  help="name of the model to be tuned this name will be used to save the model")


    ## data_params
    p.add_argument("--data_path", default="dataset/GOOG-year_small.csv",  help="path of the training file")



    args = p.parse_args()
    args = load_config(args, args.config_file, args.config_mode, verbose=0)
    return args



def data_loader(data_params):
    if data_params["data_type"] == "pandas" :
      df = pd.read_csv(data_params["data_path"])

    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

    minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype('float32'))
    df_log = minmax.transform(df.iloc[:, 1:].astype('float32'))
    df_log = pd.DataFrame(df_log)
    return df_log



def test_all():
    df_log = data_loader()
    pars =  {
        "learning_rate": {"type": "log_uniform", "init": 0.01,  "range" :(0.001, 0.1)},
        "num_layers":    {"type": "int", "init": 2,  "range" :(2, 4)},
        "size_layer":    {"type" : 'categorical', "value": [128, 256 ] },
        "timestep":      {"type" : 'categorical', "value": [5] },
        "epoch":        {"type" : 'categorical', "value": [5] },
    }
    res = optim('model_tf.1_lstm', pars=pars, df = df_log,ntrials=7 )
    print(res)


def test_fast():
    df_log = data_loader('dataset/GOOG-year_small.csv')
    pars =  {
        "learning_rate": {"type": "log_uniform", "init": 0.01,  "range" :(0.001, 0.1)},
        "num_layers":    {"type": "int", "init": 2,  "range" :(2, 4)},
        "size_layer":    {"type" : 'categorical', "value": [128, 256 ] },
        "timestep":      {"type" : 'categorical', "value": [5] },
        "epoch":         {"type" : 'categorical', "value": [2] },
    }

    res = optim('model_tf.1_lstm', pars=pars, df = df_log,ntrials=3,
                optim_params={"method" : "prune", )
    print("\n#############  Finished OPTIMIZATION  ###############")
    print(res)



if __name__ == "__main__":
    #test_all() # tot test all te modules inside model_tf
    arg = load_arguments()

    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    if arg.do == "test"  :
        test_fast()

    if arg.do == "test_all"  :
        test_all()


    if arg.do == "search"  :
        js = json.load(open(arg.config_file, 'r'))  #Config

        model_params = js.get(["model_params"])
        data_params = js.get(["data_params"])
        optim_params = js.get(["optim_params"])

        model_params  = {} if model_params is None
        data_params  = { "data_path" : arg.data_path, "data_type": "pandas" } if data_params is None
        optim_params = { 'engine': "optuna", "method" : 'prune' }  if optim_params is None

        res = optim(arg.modelname,
                    model_params,
                    ntrials=int(arg.ntrials),
                    optim_params= optim_params,
                    data_params= data_params,
                    save_folder= arg.save_folder)  # '1_lstm'

        print("#############  Finished OPTIMIZATION  ###############")
        print(res)






