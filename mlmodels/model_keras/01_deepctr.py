""""

https://github.com/shenweichen/DeepCTR


# DeepCTR

https://deepctr-doc.readthedocs.io/en/latest/Examples.html#classification-criteo


DeepCTR is a **Easy-to-use**,**Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layers which can be used to easily build custom models.It is compatible with **tensorflow 1.4+ and 2.0+**.You can use any complex model with `model.fit()`and `model.predict()` .

Let's [**Get Started!**](https://deepctr-doc.readthedocs.io/en/latest/Quick-Start.html)([Chinese Introduction](https://zhuanlan.zhihu.com/p/53231955))


## Models List

|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  Convolutional Click Prediction Model  | [CIKM 2015][A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)             |
| Factorization-supported Neural Network | [ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                    |
|      Product-based Neural Network      | [ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                   |
|              Wide & Deep               | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                 |
|                 DeepFM                 | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                           |
|        Piece-wise Linear Model         | [arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)                                 |
|          Deep & Cross Network          | [ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   |
|   Attentional Factorization Machine    | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |
|      Neural Factorization Machine      | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |
|                xDeepFM                 | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                         |
|                AutoInt                 | [arxiv 2018][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                              |
|         Deep Interest Network          | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)                                                       |
|    Deep Interest Evolution Network     | [AAAI 2019][Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                            |
|                  NFFM                  | [arxiv 2019][Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                |
|                 FGCNN                  | [WWW 2019][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction ](https://arxiv.org/pdf/1904.04447)                             |
|     Deep Session Interest Network      | [IJCAI 2019][Deep Session Interest Network for Click-Through Rate Prediction ](https://arxiv.org/abs/1905.06482)                                                |
|                FiBiNET                 | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)   |



"""
import os 

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat,get_feature_names







"""


    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.do simple Transformation for dense features
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.set hashing space for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=1000,embedding_dim=4, use_hash=True, dtype='string')  # since the input is string
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                          for feat in dense_features]

    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)

    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}


    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns,dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))





"""



####################################################################################################
# Helper functions
def os_package_root_path(filepath, sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    path = Path(os.path.realpath(filepath)).parent
    print("path: ", path)
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)


####################################################################################################
class Model():
    def __init__(self, model_pars=None, compute_pars=None, **kwargs):
        # 4.Define Model,train,predict and evaluate
        _, linear_feature_columns, dnn_feature_columns, _, _ = kwargs.get('dataset')
        
        self.model = DeepFM(linear_feature_columns,dnn_feature_columns, task='binary')
        self.model.compile(model_pars["optimization"], model_pars["cost"],
                      metrics=['binary_crossentropy'], )


####################################################################################################
def get_dataset(**kw):
    ##check whether dataset is of kind train or test
    data_path = kw['train_data_path']

    #### read from csv file
    if  kw.get("uri_type") == "pickle" :
        data = pd.read_pickle(data_path)
    else :
        data = pd.read_csv(data_path)
        sparse_features = ['C' + str(i) for i in range(1, 27)]
        dense_features = ['I' + str(i) for i in range(1, 14)]
        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )

        # Transformation for dense features
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])
        sparse_features = ['C' + str(i) for i in range(1, 27)]
        dense_features = ['I' + str(i) for i in range(1, 14)]
        
        # set hashing space for each sparse field,and record dense feature field name
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=1000,embedding_dim=4, use_hash=True, dtype='string')  # since the input is string
                                  for feat in sparse_features] + [DenseFeat(feat, 1, )
                              for feat in dense_features]
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )
        
        train, test = train_test_split(data, test_size=0.2)

    return data, linear_feature_columns, dnn_feature_columns, train, test


def fit(model, session=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None, **kwargs):
    ##loading dataset
    """
          Classe Model --> model,   model.model contains thte sub-model

    """
    print("data_pars: ", data_pars)
    data, linear_feature_columns, dnn_feature_columns, train, test = get_dataset(**data_pars)
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    target = ['label']
    model.model.fit(train_model_input, train[target].values,
                        batch_size=compute_pars['batch_size'], epochs=compute_pars['epochs'], verbose=2, 
                          validation_split=compute_pars['validation_split'], )
    
    return model


# Model p redict
def predict(model, data_pars, compute_pars=None, out_pars=None, **kwargs):
    ##  Model is class
    ## load test dataset
    data, linear_feature_columns, dnn_feature_columns, train, test=get_dataset(**data_pars)
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )
    test_model_input = {name:test[name] for name in feature_names}

    ## predict
    pred_ans = model.model.predict(test_model_input, batch_size=256)

    ### output stats for forecast entry
#     if VERBOSE:
#          pass
        
    return pred_ans


def metrics(ypred, data_pars, compute_pars=None, out_pars=None, **kwargs):
    ## load test dataset
    _, linear_feature_columns, dnn_feature_columns, _, test = get_dataset(**data_pars)
    target = ['label']
    
    metrics_dict = {"LogLoss": round(log_loss(test[target].values, ypred), 4),
                   "AUC": round(roc_auc_score(test[target].values, ypred), 4)}
    return metrics_dict



def save(model, path):
    if os.path.exists(path):
        print("exist")


def load(path):
    if os.path.exists(path):
        print("exist")
    model = Model_empty()
    model.model = predictor_deserialized
    #### Add back the model parameters...


    return model




########################################################################################################################
def get_params(choice=0, data_path="dataset/", **kw) :
    if choice == 0 :
        log("#### Path params   ################################################")
        data_path = os_package_root_path(__file__, sublevel=1, path_add=data_path)
        out_path = os.getcwd() + "/deepctr_test/"
        os.makedirs(out_path, exist_ok=True)
        log(data_path, out_path)

        train_data_path = data_path + "criteo_sample.txt"
        data_pars = {"train_data_path": train_data_path}

        log("#### Model params   ################################################")
        model_pars = {"optimization": "adam", "cost": "binary_crossentropy"}

        compute_pars = {"batch_size": 256, "epochs": 10, "validation_split": 0.2}

        out_pars = {"plot_prob": True, "quantiles": [0.1, 0.5, 0.9]}
        out_pars["path"] = data_path + out_path

    return model_pars, data_pars, compute_pars, out_pars




########################################################################################################################
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
    print(model_pars, data_pars, compute_pars, out_pars)


    log("#### Loading dataset   #############################################")
    dataset = get_dataset(**data_pars)


    log("#### Model init, fit   #############################################")
    model = Model(model_pars=model_pars, compute_pars=compute_pars, dataset=dataset)
    #model=m.model    ### WE WORK WITH THE CLASS (not the attribute GLUON )
    model=fit(model, data_pars=data_pars, model_pars=model_pars, compute_pars=compute_pars)

    log("#### Predict   ####################################################")
    
    ypred = predict(model, data_pars, compute_pars, out_pars)

    log("#### metrics   ####################################################")
    metrics_val = metrics(ypred, data_pars, compute_pars, out_pars)
    print(metrics_val)


    log("#### Plot   #######################################################")
#     plot_prob_forecasts(ypred, metrics_val, out_pars)
#     plot_predict(ypred, metrics_val, out_pars)



if __name__ == '__main__':
    VERBOSE = True
    test()
 