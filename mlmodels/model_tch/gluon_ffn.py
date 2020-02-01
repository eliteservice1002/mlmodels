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
VERBOSE=True
CHECKPOINT_NAME="GlUON"


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


def get_dataset(**kwargs):
    ##check whether dataset is of kind train or test
    TRAIN=kwargs['train']
    if TRAIN:
        data_path = kwargs['train_data_path']
    else:
        data_path = kwargs['test_data_path']

    ####read from csv file
    data_set = pd.read_csv(data_path)
    start=kwargs['start']
    num_series=kwargs['num_series']
    ### convert to gluont format
    gluonts_ds = ListDataset([{FieldName.TARGET: data_set.iloc[i].values,
                            FieldName.START: start}
                           for i in range(num_series)],
                             freq=kwargs['freq'])
    if VERBOSE:
        entry = next(iter(gluonts_ds))
        train_series = to_pandas(entry)
        train_series.plot()
        save_fig = kwargs['save_fig']
        plt.savefig(save_fig)
    return gluonts_ds






######################################################################################################
# Model fit

def fit(data_pars,model_pars, compute_pars={}, out_pars=None, **kwargs):
    ##loading dataset
    gluont_ds = get_dataset(**data_pars)

    ## load trainer
    ctx=compute_pars['ctx']
    epochs=compute_pars['epochs']
    learning_rate=compute_pars['learning_rate']
    hybridize=compute_pars['hybridize']
    num_batches_per_epoch=compute_pars['num_batches_per_epoch']
    trainer = Trainer(ctx=ctx,epochs=epochs,learning_rate=learning_rate,
                      hybridize=hybridize,num_batches_per_epoch=num_batches_per_epoch)

    ##set up the model
    num_hidden_dimensions=model_pars['num_hidden_dimensions']
    prediction_length=model_pars['prediction_length']
    context_length=model_pars['context_length']
    freq=model_pars['freq']
    estimator = SimpleFeedForwardEstimator(num_hidden_dimensions=num_hidden_dimensions,
                                           prediction_length=prediction_length,
                                           context_length=context_length,
                                           freq=freq,trainer=trainer)




    # fit the model
    predictor = estimator.train(gluont_ds)

    # save the model
    save(predictor)



###############################################################################################################
# Model predict
def predict(data_pars, compute_pars={}, out_pars=None, **kwargs):
    ## load test dataset
    data_pars['train']=False
    test_ds=get_dataset(**data_pars)

    ##load model
    predictor=load()

    num_samples=compute_pars['num_samples']


    ## make evlauation
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=num_samples,  # number of sample paths we want for evaluation
    )

    ##convert generator to list
    forecasts = list(forecast_it)
    tss = list(ts_it)


    forecast_entry = forecasts[0]
    ts_entry=tss[0]

    ### output stats for forecast entry
    if VERBOSE:
        print(f"Number of sample paths: {forecast_entry.num_samples}")
        print(f"Dimension of samples: {forecast_entry.samples.shape}")
        print(f"Start date of the forecast window: {forecast_entry.start_date}")
        print(f"Frequency of the time series: {forecast_entry.freq}")
        print(f"Mean of the future window:\n {forecast_entry.mean}")
        print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")

    #plot forecast probability

    if out_pars['plot_prob']:
        plot_prob_forecasts(ts_entry, forecast_entry)


    ## evaluate
    evaluator = Evaluator(quantiles=out_pars['quantiles'])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))

    return item_metrics


def predict_metrics(data_pars, compute_pars={}, out_pars=None, **kwargs):
    ## load test dataset
    data_pars['train']=False
    test_ds=get_dataset(**data_pars)

    ##load model
    predictor=load()

    num_samples=compute_pars['num_samples']


    ## make evlauation
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=num_samples,  # number of sample paths we want for evaluation
    )

    ##convert generator to list
    forecasts = list(forecast_it)
    tss = list(ts_it)

    ## evaluate
    evaluator = Evaluator(quantiles=out_pars['quantiles'])
    agg_metrics, _ = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
    metrics=json.dump(agg_metrics,indent=4)
    return metrics






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
def save(model):
    if os.path.exists(CHECKPOINT_NAME):
       model.serialize(Path(CHECKPOINT_NAME))


def load():
    if os.path.exists(CHECKPOINT_NAME):
        predictor_deserialized = Predictor.deserialize(Path(CHECKPOINT_NAME))
    return predictor_deserialized


def test2(data_path="dataset/", out_path="GLUON/gluon.png", reset=True):
    ###loading the command line arguments
    # arg = load_arguments()
    data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
    train_data_path = data_path + "GLUON-GLUON-train.csv"
    test_data_path = data_path + "GLUON-test.csv"
    start = pd.Timestamp("01-01-1750", freq='1H')
    data_pars = {"train_data_path": train_data_path, "test_data_path": test_data_path, "train": False,
                 'prediction_length': 48, 'freq': '1H', "start": start, "num_series": 245, "save_fig": "./series.png"}


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
    module, model = module_load_full("GlUON", model_pars)
    print(module, model)

    log("############ Model fit   ##################################")
    fit(model, module, data_pars=data_pars, out_pars=out_pars, compute_pars= compute_pars)

    log("############ Prediction  ##################################")

    agg_metrics, item_metrics = predict(model, module,data_pars=data_pars,out_pars=out_pars, compute_pars=compute_pars)

    out_pars['agg_metrics'] = agg_metrics
    out_pars['item_metrics'] = item_metrics

    ##metric
    ypred = metrics(out_pars)

    # print results
    print(ypred)
    print(item_metrics.head())



def test(data_path="dataset/"):

    ###loading the command line arguments
    data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
    print(data_path)
    train_data_path=data_path+"GLUON-GLUON-train.csv"
    test_data_path=data_path+"GLUON-test.csv"
    start=pd.Timestamp("01-01-1750", freq='1H')
    data_pars = {"train_data_path":train_data_path,"test_data_path":test_data_path,"train":False,
                 'prediction_length': 48,'freq': '1H',"start":start,"num_series":245,"save_fig":"./series.png"}

    ##loading dataset
    gluont_ds = get_dataset(**data_pars)

    ## training model

    model_pars = {"num_hidden_dimensions": [10], "prediction_length": data_pars["prediction_length"],
                  "context_length":2*data_pars["prediction_length"],"freq":data_pars["freq"]
                  }
    compute_pars={"ctx":"cpu","epochs":5,"learning_rate":1e-3,"hybridize":False,
                  "num_batches_per_epoch":100,'num_samples':100}

    fit(data_pars,model_pars, compute_pars)

    out_pars={"plot_prob":True,"quantiles":[0.1, 0.5, 0.9]}

    #### Predict
    ypred = predict(data_pars, compute_pars,out_pars)

    # print results
    print(ypred.head())

    ###predict metrics
    _, metrics = predict_metrics(data_pars, compute_pars,out_pars)


    out_pars['outpath']="GLUON/gluon.png"

    #### Plot
    plot_predict(out_pars)

if __name__ == '__main__':

    test()