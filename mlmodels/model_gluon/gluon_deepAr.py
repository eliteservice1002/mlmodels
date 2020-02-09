from gluonts.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator



from mlmodels.model_gluon.util import *


########################################################################################################################
#### Model defintion
class Model(object) :
    def __init__(self, model_pars=None, compute_pars=None) :
        ## Empty model for Seaialization
        if model_pars is None and compute_pars is None :
            self.model = None

        else :    
          self.compute_pars=compute_pars
          self.model_pars = model_pars

          m=self.compute_pars
          trainer = Trainer(batch_size=m['batch_size'],clip_gradient=m['clip_gradient'],ctx=m["ctx"], epochs=m["epochs"],
                          learning_rate=m["learning_rate"],init=m['init'],
                          learning_rate_decay_factor=m['learning_rate_decay_factor'],
                          minimum_learning_rate=m['minimum_learning_rate'],hybridize=m["hybridize"],
                          num_batches_per_epoch=m["num_batches_per_epoch"],
                          patience=m['patience'],weight_decay=m['weight_decay']
                          )

          ##set up the model
          m = self.model_pars
          self.model = DeepAREstimator(prediction_length=m['prediction_length'],freq=m['freq'],num_layers=m['num_layers'],
                                num_cells= m["num_cells"],
                                cell_type= m["cell_type"], dropout_rate=m["dropout_rate"],
                                use_feat_dynamic_real=m["use_feat_dynamic_real"],
                                use_feat_static_cat=m ['use_feat_static_cat'],use_feat_static_real=m['use_feat_static_real'],
                                scaling=m['scaling'] ,num_parallel_samples=m['num_parallel_samples'],trainer=trainer)




########################################################################################################################
def get_params(choice=0, data_path="dataset/", **kw) :
    if choice == 0 :
        log("#### Path params   ################################################")
        data_path = os_package_root_path(__file__, sublevel=2, path_add=data_path)
        out_path = os.get_cwd() + "/GLUON_test/"
        os.makedirs(out_path, exists_ok=True)
        log(data_path, out_path)

        train_data_path = data_path + "GLUON-GLUON-train.csv"
        test_data_path = data_path + "GLUON-test.csv"
        start = pd.Timestamp("01-01-1750", freq='1H')
        data_pars = {"train_data_path": train_data_path, "test_data_path": test_data_path, "train": False,
                     'prediction_length': 48, 'freq': '1H', "start": start, "num_series": 245,
                     "save_fig": "./series.png"}

        log("#### Model params   ################################################")
        model_pars = {"prediction_length": data_pars["prediction_length"], "freq": data_pars["freq"],
                      "num_layers": 2, "num_cells": 40, "cell_type": 'lstm', "dropout_rate": 0.1,
                      "use_feat_dynamic_real": False, "use_feat_static_cat": False, "use_feat_static_real": False,
                      "scaling": True, "num_parallel_samples": 100}

        compute_pars = {"batch_size": 32, "clip_gradient": 100, "ctx": None, "epochs": 1, "init": "xavier",
                        "learning_rate": 1e-3,
                        "learning_rate_decay_factor": 0.5, "hybridize": False, "num_batches_per_epoch": 100,
                        'num_samples': 100,
                        "minimum_learning_rate": 5e-05, "patience": 10, "weight_decay": 1e-08}

        out_pars = {"plot_prob": True, "quantiles": [0.1, 0.5, 0.9]}
        out_pars["path"] = data_path + out_path

    return model_pars, data_pars, compute_pars, out_pars



########################################################################################################################
def test2(data_path="dataset/", out_path="GLUON/gluon.png", reset=True):
    ###loading the command line arguments
    # arg = load_arguments()

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=0, data_path=data_path)
    model_uri = "model_gluon/gluon_deepar.py"

    log("#### Loading dataset   ############################################")
    gluont_ds = get_dataset(**data_pars)


    log("#### Model init, fit   ############################################")
    from mlmodels.models import module_load_full, fit, predict
    module, model = module_load_full(model_uri, model_pars)
    print(module, model)

    model=fit(model, None, data_pars, model_pars, compute_pars)


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


    log("#### Model init, fit   #############################################")
    model = Model(model_pars, compute_pars)
    #model=m.model    ### WE WORK WITH THE CLASS (not the attribute GLUON )
    model=fit(model, data_pars, model_pars, compute_pars)


    log("#### Predict   ####################################################")
    ypred = predict(model, data_pars, compute_pars, out_pars)
    print(ypred)


    log("#### metrics   ####################################################")
    metrics_val = metrics(ypred, data_pars, compute_pars, out_pars)


    log("#### Plot   #######################################################")
    plot_prob_forecasts(ypred, metrics_val, out_pars)
    plot_predict(ypred, metrics_val, out_pars)



if __name__ == '__main__':
    VERBOSE = True
    test()


