from gluonts.trainer import Trainer
from mlmodels.model_gluon.util import *
from gluonts.model.prophet import ProphetPredictor


######################################################################################################
#### Model defintion
class Model(object) :
    def __init__(self, model_pars=None, compute_pars=None) :
        ## Empty model for Seaialization
        if model_pars is None and compute_pars is None :
            self.model = None
        else :
            m = model_pars
            self.model = ProphetPredictor(prediction_length=m['prediction_length'],freq=m['freq'])





########################################################################################################################
def get_params(choice=0, data_path="dataset/", **kw):
    if choice == 0:
        log("#### Path params   ################################################")
        data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
        out_path = os.getcwd() + "/GLUON_prophet/"
        os.makedirs(out_path, exist_ok=True)

        model_path = os.getcwd() + "/GLUON/model_prophet/"
        os.makedirs(model_path, exist_ok=True)
        log(data_path, out_path, model_path)


        train_data_path = data_path + "GLUON-GLUON-train.csv"
        test_data_path = data_path + "GLUON-test.csv"
        start = pd.Timestamp("01-01-1750", freq='1H')
        data_pars = {"train_data_path": train_data_path, "test_data_path": test_data_path, "train": False,
                     'prediction_length': 48, 'freq': '1H', "start": start, "num_series": 245,
                     "save_fig": "./series.png","modelpath":model_path}

        log("## Model params   ################################################")
        model_pars = {"prediction_length": data_pars["prediction_length"], "freq": data_pars["freq"]}
        compute_pars = {}
        outpath = out_path + "result"

        out_pars = {"outpath": outpath, "plot_prob": True, "quantiles": [0.1, 0.5, 0.9]}

    return model_pars, data_pars, compute_pars, out_pars


########################################################################################################################
def test2(data_path="dataset/", out_path="GLUON/gluon.png", reset=True):
    ###loading the command line arguments
    # arg = load_arguments()

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=0, data_path=data_path)
    model_uri = "model_gluon/gluon_deepar.py"

    log("#### Loading dataset   #############################################")
    gluont_ds = get_dataset(**data_pars)

    log("#### Model init, fit   ###########################################")
    from mlmodels.models import module_load_full, fit, predict
    module, model = module_load_full(model_uri, model_pars)
    print(module, model)

    model = fit(model, None, data_pars, model_pars, compute_pars)

    log("#### save the trained model  #############################################")
    save(model, data_pars["modelpath"])


    log("#### Predict   ###################################################")
    ypred = predict(model, data_pars, compute_pars, out_pars)
    print(ypred)

    log("###Get  metrics   ################################################")
    metrics_val = metrics(model, data_pars, compute_pars, out_pars)


    log("#### Plot   ######################################################")
    plot_prob_forecasts(ypred, metrics_val, out_pars)
    plot_predict(ypred, metrics_val, out_pars)


def test(data_path="dataset/"):
    ### Local test

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=0, data_path=data_path)


    log("#### Loading dataset   #############################################")
    gluont_ds = get_dataset(**data_pars)


    log("#### Model init, fit   ###########################################")
    model = Model(model_pars, compute_pars)
    #model=m.model    ### WE WORK WITH THE CLASS (not the attribute GLUON )
    model=fit(model, data_pars, model_pars, compute_pars)

    log("#### save the trained model  #############################################")
    save(model, data_pars["modelpath"])


    log("#### Predict   ###################################################")
    ypred = predict(model, data_pars, compute_pars, out_pars)
    print(ypred)


    log("#### metrics   ################################################")
    metrics_val, item_metrics = metrics(ypred, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   #######################################################")
    plot_prob_forecasts(ypred, out_pars)
    plot_predict(item_metrics, out_pars)


if __name__ == '__main__':
    VERBOSE = True
    test()