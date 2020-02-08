import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.model.predictor import Predictor
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

VERBOSE = FALSE

####################################################################################################
# Helper functions
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


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)


####################################################################################################
# Dataaset
def get_dataset(**kw):
    ##check whether dataset is of kind train or test
    data_path = kw['train_data_path'] if  kw['train'] else kw['test_data_path']

    #### read from csv file
    if  kw.get("uri_type") == "pickle" :
        data_set = pd.read_pickle(data_path)
    else :
        data_set = pd.read_csv(data_path)

    ### convert to gluont format
    gluonts_ds = ListDataset([{FieldName.TARGET: data_set.iloc[i].values, FieldName.START: kw['start'] }
                             for i in range(kw['num_series'])],  freq=kw['freq'])

    if VERBOSE:
        entry = next(iter(gluonts_ds))
        train_series = to_pandas(entry)
        train_series.plot()
        save_fig = kw['save_fig']
        plt.savefig(save_fig)

    return gluonts_ds





######################################################################################################
#### Model defintion
class Model(object) :
  def __init__(self, model_pars, compute_pars) :
    ## load trainer
    m = compute_pars
    trainer = Trainer(ctx=m["ctx"], epochs=m["epochs"], learning_rate=m["learning_rate"],
                      hybridize=m["hybridize"], num_batches_per_epoch=m["num_batches_per_epoch"])

    ##set up the model
    m = model_pars
    model = SimpleFeedForwardEstimator(num_hidden_dimensions=m['prediction_length'],
                                           prediction_length= m["prediction_length"],
                                           context_length= m["context_length"],
                                           freq=m["freq"],trainer=trainer)         
    return model


# Model fit
def fit(model, data_pars, model_pars=None, compute_pars=None, out_pars=None, **kwargs):
    ##loading dataset
    gluont_ds = get_dataset( **data_pars )
    predictor = model.train( gluont_ds )
    return predictor


# Model predict
def predict(model, data_pars, compute_pars=None, out_pars=None, **kwargs):
    ## load test dataset
    data_pars['train']=False
    test_ds=get_dataset(**data_pars)


    ## predict
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=model,  # predictor
        num_samples=compute_pars['num_samples'],  # number of sample paths we want for evaluation
    )

    ##convert generator to list
    forecasts,tss = list(forecast_it), list(ts_it)
    forecast_entry, ts_entry = forecasts[0], tss[0] 

    ### output stats for forecast entry
    if VERBOSE:
        print(f"Number of sample paths: {forecast_entry.num_samples}")
        print(f"Dimension of samples: {forecast_entry.samples.shape}")
        print(f"Start date of the forecast window: {forecast_entry.start_date}")
        print(f"Frequency of the time series: {forecast_entry.freq}")
        print(f"Mean of the future window:\n {forecast_entry.mean}")
        print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")

    dd = { "forecasts": forecasts, "tss" :tss    } 
    return dd


def metrics(ypred, data_pars, compute_pars=None, out_pars=None, **kwargs):
    ## load test dataset
    data_pars['train'] = False
    test_ds = get_dataset(**data_pars)
    path_model = compute_pars["path"]
    
    
    forecasts = ypred["forecast_it"]  
    tss =  ypred["ts_it"]  
    
    ## evaluate
    evaluator = Evaluator(quantiles=out_pars['quantiles'])
    agg_metrics, _ = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
    metrics_dict=json.dump(agg_metrics,indent=4)
    return metrics_dict


###############################################################################################################
### different plots and output metric
def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()


def plot_predict(out_pars=None):
    item_metrics=out_pars['item_metrics']
    item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
    plt.grid(which="both")
    outpath=out_pars['outpath']
    plt.savefig(outpath)
    plt.clf()
    print('Saved image to {}.'.format(outpath))



###############################################################################################################
# save and load model helper function
def save(model, path):
    if os.path.exists(path):
       model.serialize(Path(path))


def load(path):
    if os.path.exists( path ):
        predictor_deserialized = Predictor.deserialize(Path( path ))
    return predictor_deserialized


def test2(data_path="dataset/", out_path="GLUON/gluon.png", reset=True):
    ###loading the command line arguments
    # arg = load_arguments()
    data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
    out_path = os.get_cwd() + "/GLUON/"
    os.makedirs(out_path, exists_ok=True)
    log(data_path, out_path)


    train_data_path = data_path + "GLUON-GLUON-train.csv"
    test_data_path = data_path + "GLUON-test.csv"
    start = pd.Timestamp("01-01-1750", freq='1H')
    data_pars = {"train_data_path": train_data_path, "test_data_path": test_data_path, "train": False,
                 'prediction_length': 48, 'freq': '1H', "start": start, "num_series": 245,
                 "save_fig": "./series.png"}

    ##loading dataset
    gluont_ds = get_dataset(**data_pars)

    ##Params
    model_pars = {"num_hidden_dimensions": [10], "prediction_length": data_pars["prediction_length"],
                  "context_length": 2 * data_pars["prediction_length"], "freq": data_pars["freq"]
                  }
    compute_pars = {"ctx": "cpu", "epochs": 5, "learning_rate": 1e-3, "hybridize": False,
                    "num_batches_per_epoch": 100, 'num_samples': 100}


    out_pars = {"plot_prob": True, "quantiles": [0.1, 0.5, 0.9]}
    out_pars["path"]= data_path + out_path


    log("############ Model preparation   #########################")
    from mlmodels.models import module_load_full, fit, predict
    module, model = module_load_full("model_gluon/gluon_ffn.py", model_pars)
    print(module, model)

    log("#### Predict   ###################################################")
    ypred = predict(model, data_pars, compute_pars, out_pars)
    print(ypred)

    log("###Get  metrics   ################################################")
    metrics_val = metrics(model, data_pars, compute_pars, out_pars)

    log("#### Plot   ######################################################")
    forecast_entry = ypred["forecast"][0]
    ts_entry = ypred["tss"][0]
    plot_prob_forecasts(ts_entry, forecast_entry)
    plot_predict(out_pars)



def test(data_path="dataset/"):
    ###loading the command line arguments
    data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
    out_path = os.getcwd() + "/GLUON/"
    os.makedirs(out_path, exist_ok= True)
    log(data_path, out_path)
   
    train_data_path = data_path+"GLUON-GLUON-train.csv"
    test_data_path  = data_path+"GLUON-test.csv"
    start = pd.Timestamp("01-01-1750", freq='1H')
    data_pars = {"train_data_path":train_data_path,"test_data_path":test_data_path,"train":False,
                 'prediction_length': 48,'freq': '1H',"start":start,"num_series":245,
                 "save_fig":"./series.png"}


    log("##loading dataset   ##############################################")
    gluont_ds = get_dataset(**data_pars)

    
    log("## Model params   ################################################")
    model_pars = {"num_hidden_dimensions": [10], "prediction_length": data_pars["prediction_length"],
                  "context_length":2*data_pars["prediction_length"],"freq":data_pars["freq"]
                 }
    compute_pars = {"ctx":"cpu","epochs":5,"learning_rate":1e-3,"hybridize":False,
                  "num_batches_per_epoch":100,'num_samples':100}

    out_pars = {"outpath": out_path, "plot_prob": True, "quantiles": [0.1, 0.5, 0.9]}


    log("#### Model init, fit   ###########################################")
    model = Model( model_pars, compute_pars ) 
    model = fit(model, data_pars, model_pars, compute_pars)


    log("#### Predict   ###################################################")
    ypred    = predict(model, data_pars, compute_pars, out_pars)
    print(ypred)

            
    log("###Get  metrics   ################################################")
    metrics_val = metrics(model, data_pars, compute_pars,out_pars)
    

    log("#### Plot   ######################################################")
    forecast_entry = ypred["forecast"][0]
    ts_entry = ypred["tss"][0]
    plot_prob_forecasts(ts_entry, forecast_entry)
    plot_predict(out_pars)
                        


                        
if __name__ == '__main__':
    VERBOSE=True
    test()
    
    
    
    
    
    
 
