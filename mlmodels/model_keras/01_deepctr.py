""""
# DeepCTR
https://github.com/shenweichen/DeepCTR
https://deepctr-doc.readthedocs.io/en/latest/Examples.html#classification-criteo


DeepCTR is a **Easy-to-use**,**Modular** and **Extendible** package of deep-learning based CTR models 
along with lots of core components layers which can be used to easily build custom models.It is compatible with **tensorflow 1.4+ and 2.0+**.You can use any complex model with `model.fit()`and `model.predict()` .


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

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from tensorflow.python.keras.preprocessing.sequence import pad_sequences


from deepctr.inputs import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM


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
        _, linear_cols, dnn_cols, _, _, _ = kwargs.get('dataset')

        self.model = DeepFM(linear_cols, dnn_cols, task=compute_pars['task'])
        self.model.compile(model_pars["optimization"], model_pars["cost"],
                           metrics=['binary_crossentropy'], )


####################################################################################################
def get_dataset(**kw):
    ##check whether dataset is of kind train or test
    data_path = kw['train_data_path']

    #### read from csv file
    if kw.get("uri_type") == "pickle":
        data = pd.read_pickle(data_path)
        target = ""
    else:
        data = pd.read_csv(data_path)
        if "criteo_sample.txt" in data_path:
            hash_feature = kw.get('hash_feature')
            sparse_col = ['C' + str(i) for i in range(1, 27)]
            dense_col = ['I' + str(i) for i in range(1, 14)]
            data[sparse_col] = data[sparse_col].fillna('-1', )
            data[dense_col] = data[dense_col].fillna(0, )
            target = ["label"]

            # set hashing space for each sparse field,and record dense feature field name
            if hash_feature:
                # Transformation for dense features
                mms = MinMaxScaler(feature_range=(0, 1))
                data[dense_col] = mms.fit_transform(data[dense_col])
                sparse_col = ['C' + str(i) for i in range(1, 27)]
                dense_col = ['I' + str(i) for i in range(1, 14)]

                fixlen_cols = [SparseFeat(feat, vocabulary_size=1000, embedding_dim=4, use_hash=True,
                                                     dtype='string')  # since the input is string
                                          for feat in sparse_col] + [DenseFeat(feat, 1, )
                                                                          for feat in dense_col]
            else:
                for feat in sparse_col:
                    lbe = LabelEncoder()
                    data[feat] = lbe.fit_transform(data[feat])
                mms = MinMaxScaler(feature_range=(0, 1))
                data[dense_col] = mms.fit_transform(data[dense_col])
                fixlen_cols = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                                          for i, feat in enumerate(sparse_col)] + [DenseFeat(feat, 1, )
                                                                                        for feat in dense_col]
            linear_cols = fixlen_cols
            dnn_cols = fixlen_cols

            train, test = train_test_split(data, test_size=0.2)
        elif "movielens_sample.txt" in data_path:
            multiple_value = kw.get('multiple_value')
            sparse_col = ["movie_id", "user_id",
                               "gender", "age", "occupation", "zip"]
            target = ['rating']
            # 1.Label Encoding for sparse features,and do simple Transformation for dense features
            for feat in sparse_col:
                lbe = LabelEncoder()
                data[feat] = lbe.fit_transform(data[feat])
            if not multiple_value:
                # 2.count #unique features for each sparse field
                fixlen_cols = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                                          for feat in sparse_col]
                linear_cols = fixlen_cols
                dnn_cols = fixlen_cols

                train, test = train_test_split(data, test_size=0.2)
            else:
                hash_feature = kw.get('hash_feature', False)
                if not hash_feature:
                    def split(x):
                        key_ans = x.split('|')
                        for key in key_ans:
                            if key not in key2index:
                                # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
                                key2index[key] = len(key2index) + 1
                        return list(map(lambda x: key2index[x], key_ans))

                    # preprocess the sequence feature
                    key2index = {}
                    genres_list = list(map(split, data['genres'].values))
                    genres_length = np.array(list(map(len, genres_list)))
                    max_len = max(genres_length)
                    # Notice : padding=`post`
                    genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )

                    fixlen_cols = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                                              for feat in sparse_col]

                    use_weighted_sequence = False
                    if use_weighted_sequence:
                        varlen_cols = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
                            key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                                                   weight_name='genres_weight')]  # Notice : value 0 is for padding for sequence input feature
                    else:
                        varlen_cols = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
                            key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                                                   weight_name=None)]  # Notice : value 0 is for padding for sequence input feature

                    linear_cols = fixlen_cols + varlen_cols
                    dnn_cols = fixlen_cols + varlen_cols

                    # generate input data for model
                    model_input = {name: data[name] for name in sparse_col}  #
                    model_input["genres"] = genres_list
                    model_input["genres_weight"] = np.random.randn(data.shape[0], max_len, 1)
                else:
                    data[sparse_col] = data[sparse_col].astype(str)
                    # 1.Use hashing encoding on the fly for sparse features,and process sequence features
                    genres_list = list(map(lambda x: x.split('|'), data['genres'].values))
                    genres_length = np.array(list(map(len, genres_list)))
                    max_len = max(genres_length)

                    # Notice : padding=`post`
                    genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', dtype=str, value=0)

                    # 2.set hashing space for each sparse field and generate feature config for sequence feature

                    fixlen_cols = [
                        SparseFeat(feat, data[feat].nunique() * 5, embedding_dim=4, use_hash=True, dtype='string')
                        for feat in sparse_col]
                    varlen_cols = [
                        VarLenSparseFeat(
                            SparseFeat('genres', vocabulary_size=100, embedding_dim=4, use_hash=True, dtype="string"),
                            maxlen=max_len, combiner='mean',
                        )]  # Notice : value 0 is for padding for sequence input feature
                    linear_cols = fixlen_cols + varlen_cols
                    dnn_cols = fixlen_cols + varlen_cols
                    feature_names = get_feature_names(linear_cols + dnn_cols)

                    # 3.generate input data for model
                    model_input = {name: data[name] for name in feature_names}
                    model_input['genres'] = genres_list

                train, test = model_input, model_input

    return data, linear_cols, dnn_cols, train, test, target


def fit(model, session=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None, **kwargs):
    ##loading dataset
    """
          Classe Model --> model,   model.model contains thte sub-model

    """
    data, linear_cols, dnn_cols, train, test, target = get_dataset(**data_pars)

    multiple_value = data_pars.get('multiple_value', None)
    if multiple_value is None:
        feature_names = get_feature_names(linear_cols + dnn_cols, )
        train_model_input = {name: train[name] for name in feature_names}
        model.model.fit(train_model_input, train[target].values,
                        batch_size=compute_pars['batch_size'], epochs=compute_pars['epochs'], verbose=2,
                        validation_split=compute_pars['validation_split'], )
    else:
        model.model.fit(train, data[target].values,
                        batch_size=compute_pars['batch_size'], epochs=compute_pars['epochs'], verbose=2,
                        validation_split=compute_pars['validation_split'], )

    return model


# Model p redict
def predict(model, data_pars, compute_pars=None, out_pars=None, **kwargs):
    ##  Model is class
    ## load test dataset
    data, linear_cols, dnn_cols, train, test, target = get_dataset(**data_pars)
    feature_names = get_feature_names(linear_cols + dnn_cols, )
    test_model_input = {name: test[name] for name in feature_names}

    multiple_value = data_pars.get('multiple_value', None)
    ## predict
    if multiple_value is None:
        pred_ans = model.model.predict(test_model_input, batch_size=256)
    else:
        pred_ans = None

        ### output stats for forecast entry
    #     if VERBOSE:
    #          pass

    return pred_ans


def metrics(ypred, data_pars, compute_pars=None, out_pars=None, **kwargs):
    ## load test dataset
    _, linear_cols, dnn_cols, _, test, target = get_dataset(**data_pars)

    if compute_pars.get("task") == "binary":
        metrics_dict = {"LogLoss": round(log_loss(test[target].values, ypred), 4),
                        "AUC": round(roc_auc_score(test[target].values, ypred), 4)}
    elif compute_pars.get("task") == "regression":
        multiple_value = data_pars.get('multiple_value', None)
        if multiple_value is None:
            metrics_dict = {"MSE": round(mean_squared_error(test[target].values, ypred), 4)}
        else:
            metrics_dict = {}
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
def path_setup(out_folder="", sublevel=1, data_path="dataset/"):
        data_path = os_package_root_path(__file__, sublevel=sublevel, path_add=data_path)
        out_path = os.getcwd() + "/" + out_folder
        os.makedirs(out_path, exist_ok=True)
        log(data_path, out_path)
        return data_path, out_path


def get_params(choice=0, data_path="dataset/", **kw):
    if choice == 0:
        log("#### Path params   ################################################")
        data_path, out_path = path_setup(out_folder="/deepctr_test/", data_path=data_path) 

        train_data_path = data_path + "criteo_sample.txt"
        data_pars = {"train_data_path": train_data_path}

        log("#### Model params   ################################################")
        model_pars = {"optimization": "adam", "cost": "binary_crossentropy"}
        compute_pars = {"task": "binary", "batch_size": 256, "epochs": 10, "validation_split": 0.2}
        out_pars = {"plot_prob": True, "quantiles": [0.1, 0.5, 0.9], "path" : out_path  }


    elif choice == 1:
        log("#### Path params   ################################################")
        data_path, out_path = path_setup(out_folder="/deepctr_test/", data_path=data_path) 

        train_data_path = data_path + "criteo_sample.txt"
        data_pars = {"train_data_path": train_data_path, "hash_feature": True}

        log("#### Model params   ################################################")
        model_pars = {"optimization": "adam", "cost": "binary_crossentropy"}
        compute_pars = {"task": "binary", "batch_size": 256, "epochs": 10, "validation_split": 0.2}
        out_pars = {"plot_prob": True, "quantiles": [0.1, 0.5, 0.9], "path" : out_path  }


    elif choice == 2:
        log("#### Path params   ################################################")
        data_path, out_path = path_setup(out_folder="/deepctr_test/", data_path=data_path) 

        train_data_path = data_path + "movielens_sample.txt"
        data_pars = {"train_data_path": train_data_path}

        log("#### Model params   ################################################")
        model_pars = {"optimization": "adam", "cost": "mse"}
        compute_pars = {"task": "regression", "batch_size": 256, "epochs": 10, "validation_split": 0.2}
        out_pars = {"plot_prob": True, "quantiles": [0.1, 0.5, 0.9], "path" : out_path  }


    elif choice == 3:
        log("#### Path params   ################################################")
        data_path, out_path = path_setup(out_folder="/deepctr_test/", data_path=data_path) 

        train_data_path = data_path + "movielens_sample.txt"
        data_pars = {"train_data_path": train_data_path, "multiple_value": True}

        log("#### Model params   ################################################")
        model_pars = {"optimization": "adam", "cost": "mse"}
        compute_pars = {"task": "regression", "batch_size": 256, "epochs": 10, "validation_split": 0.2}
        out_pars = {"plot_prob": True, "quantiles": [0.1, 0.5, 0.9], "path" : out_path  }


    elif choice == 4:
        log("#### Path params   ################################################")
        data_path, out_path = path_setup(out_folder="/deepctr_test/", data_path=data_path) 

        train_data_path = data_path + "movielens_sample.txt"
        data_pars = {"train_data_path": train_data_path, "multiple_value": True, "hash_feature": True}

        log("#### Model params   ################################################")
        model_pars = {"optimization": "adam", "cost": "mse"}
        compute_pars = {"task": "regression", "batch_size": 256, "epochs": 10, "validation_split": 0.2}
        out_pars = {"plot_prob": True, "quantiles": [0.1, 0.5, 0.9], "path" : out_path  }

    return model_pars, data_pars, compute_pars, out_pars


########################################################################################################################
########################################################################################################################
def test(data_path="dataset/", params_choice=0):
    ### Local test

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=params_choice, data_path=data_path)
    print(model_pars, data_pars, compute_pars, out_pars)

    log("#### Loading dataset   #############################################")
    dataset = get_dataset(**data_pars)

    log("#### Model init, fit   #############################################")
    model = Model(model_pars=model_pars, compute_pars=compute_pars, dataset=dataset)
    # model=m.model    ### WE WORK WITH THE CLASS (not the attribute GLUON )
    model = fit(model, data_pars=data_pars, model_pars=model_pars, compute_pars=compute_pars)

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
    test(0)
    test(1)
    test(2)
    test(3)
    test(4)













"""


    data = pd.read_csv('./criteo_sample.txt')

    sparse_col = ['C' + str(i) for i in range(1, 27)]
    dense_col = ['I' + str(i) for i in range(1, 14)]

    data[sparse_col] = data[sparse_col].fillna('-1', )
    data[dense_col] = data[dense_col].fillna(0, )
    target = ['label']

    # 1.do simple Transformation for dense features
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_col] = mms.fit_transform(data[dense_col])

    # 2.set hashing space for each sparse field,and record dense feature field name

    fixlen_cols = [SparseFeat(feat, vocabulary_size=1000,embedding_dim=4, use_hash=True, dtype='string')  # since the input is string
                              for feat in sparse_col] + [DenseFeat(feat, 1, )
                          for feat in dense_col]

    linear_cols = fixlen_cols
    dnn_cols = fixlen_cols
    feature_names = get_feature_names(linear_cols + dnn_cols, )

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)

    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}


    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_cols,dnn_cols, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))





"""
