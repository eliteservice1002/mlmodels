
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
from mlmodels.model_gluon.util import *

VERBOSE = False





########################################################################################################################
#### Model defintion
class Model(object) :
    def __init__(self, model_pars=None, compute_pars=None) :
        ## Empty model for Seaialization
        if model_pars is None and compute_pars is None :
           self.model = None
        else :
           m = compute_pars
           trainer = Trainer(ctx=m["ctx"], epochs=m["epochs"], learning_rate=m["learning_rate"],
                      hybridize=m["hybridize"], num_batches_per_epoch=m["num_batches_per_epoch"])

           ##set up the model
           m = model_pars
           self.model = SimpleFeedForwardEstimator(num_hidden_dimensions=m['num_hidden_dimensions'],
                                           prediction_length= m["prediction_length"],
                                           context_length= m["context_length"],
                                           freq=m["freq"],trainer=trainer)





########################################################################################################################
def get_params(choice=0, data_path="dataset/", **kw):
    if choice == 0:
        log("#### Path params   ################################################")
        data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
        out_path = os.getcwd() + "/GLUON_ffn/"
        os.makedirs(out_path, exist_ok=True)

        model_path = os.getcwd() + "/GLUON/model_ffn/"
        os.makedirs(model_path, exist_ok=True)
        log(data_path, out_path,model_path)


        train_data_path = data_path + "GLUON-GLUON-train.csv"
        test_data_path = data_path + "GLUON-test.csv"
        start = pd.Timestamp("01-01-1750", freq='1H')

        data_pars = {"train_data_path": train_data_path, "test_data_path": test_data_path, "train": False,
                     'prediction_length': 48, 'freq': '1H', "start": start, "num_series": 245,
                     "save_fig": "./series.png","modelpath":model_path}


        log("#### Model params   ###################################################")
        model_pars = {"num_hidden_dimensions": [10], "prediction_length": data_pars["prediction_length"],
                      "context_length":2*data_pars["prediction_length"],"freq":data_pars["freq"]
                     }

        compute_pars = {"ctx":"cpu","epochs":5,"learning_rate":1e-3,"hybridize":False,
                      "num_batches_per_epoch":100,'num_samples':100}

        outpath=out_path+"result"

        out_pars = {"outpath": outpath, "plot_prob": True, "quantiles": [0.1, 0.5, 0.9]}

    return model_pars, data_pars, compute_pars, out_pars




def test2(data_path="dataset/", out_path="GLUON/gluon.png", reset=True):
    ###loading the command line arguments
    # arg = load_arguments()

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=0, data_path=data_path)
    model_uri = "model_gluon/gluon_ffn.py"


    log("#### Loading dataset   #############################################")
    gluont_ds = get_dataset(**data_pars)


    log("#### Model init, fit   ###########################################")
    from mlmodels.models import module_load_full, fit, predict
    module, model = module_load_full(model_uri, model_pars)
    print(module, model)

    model=fit(model, None, data_pars, model_pars, compute_pars)

    log("#### save the trained model  #############################################")
    save(model, data_pars["modelpath"])


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
    VERBOSE=True
    test()
    
    
    
    
    
    
 
