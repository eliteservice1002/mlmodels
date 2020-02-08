import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.model.predictor import Predictor
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

VERBOSE = False







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


# Model fit
def fit(model, data_pars, model_pars=None, compute_pars=None, out_pars=None, **kwargs):
    ##loading dataset
    gluont_ds = get_dataset(**data_pars)
    predictor = model.train(gluont_ds)
    return predictor


# Model predict
def predict(model, data_pars, compute_pars=None, out_pars=None, **kwargs):
    ## load test dataset
    data_pars['train'] = False
    test_ds = get_dataset(**data_pars)

    ## predict
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=model,  # predictor
        num_samples=compute_pars['num_samples'],  # number of sample paths we want for evaluation
    )

    ##convert generator to list
    forecasts, tss = list(forecast_it), list(ts_it)
    forecast_entry, ts_entry = forecasts[0], tss[0]

    ### output stats for forecast entry
    if VERBOSE:
        print(f"Number of sample paths: {forecast_entry.num_samples}")
        print(f"Dimension of samples: {forecast_entry.samples.shape}")
        print(f"Start date of the forecast window: {forecast_entry.start_date}")
        print(f"Frequency of the time series: {forecast_entry.freq}")
        print(f"Mean of the future window:\n {forecast_entry.mean}")
        print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")

    dd = {"forecasts": forecasts, "tss": tss}
    return dd


def metrics(ypred, data_pars, compute_pars=None, out_pars=None, **kwargs):
    ## load test dataset
    data_pars['train'] = False
    test_ds = get_dataset(**data_pars)


    forecasts = ypred["forecasts"]
    tss = ypred["tss"]

    ## evaluate
    evaluator = Evaluator(quantiles=out_pars['quantiles'])
    agg_metrics, _ = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
    metrics_dict = json.dumps(agg_metrics, indent=4)
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
    item_metrics = out_pars['item_metrics']
    item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
    plt.grid(which="both")
    outpath = out_pars['outpath']
    plt.savefig(outpath)
    plt.clf()
    print('Saved image to {}.'.format(outpath))


###############################################################################################################
# save and load model helper function
def save(model, path):
    if os.path.exists(path):
        model.serialize(Path(path))


def load(path):
    if os.path.exists(path):
        predictor_deserialized = Predictor.deserialize(Path(path))
    return predictor_deserialized


   # Model fit
def fit(model,data_pars, model_pars=None, compute_pars=None, out_pars=None, **kwargs):
        ##loading dataset
        gluont_ds = get_dataset( **data_pars )
        predictor = model.train( gluont_ds )
        return predictor


    # Model p redict
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
