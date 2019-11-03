#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[ ]:


### Install Requirement
get_ipython().system("pip install -r requirements.txt")


# In[98]:


get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "")
get_ipython().run_line_magic("matplotlib", "inline")
get_ipython().run_line_magic("config", "IPCompleter.greedy=True")

import gc
import logging
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

#### CatbOost
import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
### Pandas Profiling for features
# !pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
import pandas_profiling as pp
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
### MLP Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm import tqdm_notebook
from util_feature import *
from util_model import *

warnings.filterwarnings("ignore")


# In[64]:


print("ok")


# # Data Loading, basic profiling

# In[3]:


folder = os.getcwd() + "/"


# In[4]:


df = pd.read_csv(folder + "/data/address_matching_data.csv")
df.head(5)


# In[5]:


df.describe()


# In[6]:


df.columns, df.dtypes


# In[7]:


profile = df.profile_report(title="Pandas Profiling Report")
profile.to_file(output_file="output.html")


colexclude = profile.get_rejected_variables(threshold=0.98)
colexclude


# In[8]:


# In[ ]:


# # Column selection by type

# In[11]:


colid = "id"
colnum = [
    "name_levenshtein_simple",
    "name_trigram_simple",
    "name_levenshtein_term",
    "name_trigram_term",
    "city_levenshtein_simple",
    "city_trigram_simple",
    "city_levenshtein_term",
    "city_trigram_term",
    "zip_levenshtein_simple",
    "zip_trigram_simple",
    "zip_levenshtein_term",
    "zip_trigram_term",
    "street_levenshtein_simple",
    "street_trigram_simple",
    "street_levenshtein_term",
    "street_trigram_term",
    "website_levenshtein_simple",
    "website_trigram_simple",
    "website_levenshtein_term",
    "website_trigram_term",
    "phone_levenshtein",
    "phone_trigram",
    "fax_levenshtein",
    "fax_trigram",
    "street_number_levenshtein",
    "street_number_trigram",
]

colcat = ["phone_equality", "fax_equality", "street_number_equality"]
coltext = []

coldate = []

coly = "is_match"


colall = colnum + colcat + coltext

"""

dfnum, dfcat, dfnum_bin, 
dfnum_binhot,  dfcat_hot

colnum, colcat, coltext, 
colnum_bin, colnum_binhot,  

"""

print(colall)


# In[ ]:


# In[ ]:


# # Data type normalization, Encoding process (numerics, category)

# In[28]:


# Normalize to NA, NA Handling
df = df.replace("?", np.nan)


# In[29]:


### colnum procesing
for x in colnum:
    df[x] = df[x].astype("float32")

print(df.dtypes)


# In[30]:


##### Colcat processing
colcat_map = pd_colcat_mapping(df, colcat)

for col in colcat:
    df[col] = df[col].apply(lambda x: colcat_map["cat_map"][col].get(x))

print(df[colcat].dtypes, colcat_map)


# In[74]:


# # Data Distribution after encoding/ data type normalization

# In[31]:


#### ColTarget Distribution
coly_stat = pd_stat_distribution(df[["id", coly]], subsample_ratio=1.0)
coly_stat


# In[ ]:


# In[ ]:


# In[32]:


#### Col numerics distribution
colnum_stat = pd_stat_distribution(df[colnum], subsample_ratio=0.6)
colnum_stat


# In[ ]:


# In[33]:


#### Col stats distribution
colcat_stat = pd_stat_distribution(df[colcat], subsample_ratio=0.3)
colcat_stat


# In[ ]:


# In[ ]:


# # Feature processing (strategy 1)

# In[16]:


### BAcKUP data before Pre-processing
dfref = copy.deepcopy(df)
print(dfref.shape)


# In[27]:


df = copy.deepcopy(dfref)


# In[22]:


## Map numerics to Category bin
dfnum, colnum_map = pd_colnum_tocat(
    df, colname=colnum, colexclude=None, colbinmap=None, bins=5, suffix="_bin", method=""
)


print(colnum_map)


# In[37]:


colnum_bin = [x + "_bin" for x in list(colnum_map.keys())]
print(colnum_bin)


# In[38]:


dfnum[colnum_bin].head(7)


# In[39]:


### numerics bin to One Hot
dfnum_hot = pd_col_to_onehot(dfnum[colnum_bin], colname=colnum_bin, returncol=0)
colnum_hot = list(dfnum_hot.columns)
dfnum_hot.head(10)


# In[202]:





# In[40]:


dfcat_hot = pd_col_to_onehot(df[colcat], colname=colcat, returncol=0)
colcat_hot = list(dfcat_hot.columns)
dfcat_hot.head(5)


# In[ ]:


# In[ ]:


# In[ ]:


#

# # Train data preparation

# In[67]:


#### Train data preparation
dfX = pd.concat((dfnum_hot, dfcat_hot), axis=1)
colX = list(dfX.columns)
X = dfX.values
yy = df[coly].values

Xtrain, Xtest, ytrain, ytest = train_test_split(X, yy, random_state=42, test_size=0.5, shuffle=True)


print(Xtrain.shape, Xtest.shape, colX)


# In[ ]:


# In[203]:


0


# # Model evaluation

# In[ ]:


# In[42]:


### Baseline : L2 penalty to reduce overfitting
clf_log = sk.linear_model.LogisticRegression(penalty="l2", class_weight="balanced")


# In[43]:


clf_log, dd = sk_model_eval_classification(clf_log, 1, Xtrain, ytrain, Xtest, ytest)


# In[92]:


sk_model_eval_classification_cv(clf_log, X, yy, test_size=0.5, ncv=3)


# In[44]:


clf_log_feat = sk_feature_impt_logis(clf_log, colX)
clf_log_feat


# In[208]:


# In[44]:


1


# In[96]:


### Light GBM
clf_lgb = lgb.LGBMClassifier(
    learning_rate=0.125,
    metric="l2",
    max_depth=15,
    n_estimators=50,
    objective="binary",
    num_leaves=38,
    njobs=-1,
)


# In[97]:


clf_lgb, dd_lgb = sk_model_eval_classification(clf_lgb, 1, Xtrain, ytrain, Xtest, ytest)


# In[98]:


shap.initjs()

dftest = pd.DataFrame(columns=colall, data=Xtest)

explainer = shap.TreeExplainer(clf_lgb)
shap_values = explainer.shap_values(dftest)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0, :], dftest.iloc[0, :])


# In[ ]:


# visualize the training set predictions
# shap.force_plot(explainer.expected_value, shap_values, dftest)

# Plot summary_plot as barplot:
# shap.summary_plot(shap_values, Xtest, plot_type='bar')


# In[112]:


lgb_feature_imp = pd.DataFrame(
    sorted(zip(clf_lgb.feature_importances_, colall)), columns=["value", "feature"]
)
lgb_feature_imp = lgb_feature_imp.sort_values("value", ascending=0)
print(lgb_feature_imp)


plotbar(
    lgb_feature_imp.iloc[:10, :],
    colname=["value", "feature"],
    title="feature importance",
    savefile="lgb_feature_imp.png",
)


# In[118]:


kf = StratifiedKFold(n_splits=3, shuffle=True)
# partially based on https://www.kaggle.com/c0conuts/xgb-k-folds-fastai-pca
clf_list = []
for itrain, itest in kf.split(X, yy):
    print("###")
    Xtrain, Xval = X[itrain], X[itest]
    ytrain, yval = yy[itrain], yy[itest]
    clf_lgb.fit(Xtrain, ytrain, eval_set=[(Xval, yval)], early_stopping_rounds=20)

    clf_list.append(clf_lgb)


# In[122]:


for i, clfi in enumerate(clf_list):
    print(i)
    clf_lgbi, dd_lgbi = sk_model_eval_classification(clfi, 0, Xtrain, ytrain, Xtest, ytest)


# In[4]:


def np_find_indice(v, x):
    for i, j in enumerate(v):
        if j == x:
            return i
    return -1


def col_getnumpy_indice(colall, colcat):
    return [np_find_indice(colall, x) for x in colcat]


# In[7]:


colcat_idx = col_getnumpy_indice(colall, colcat)

clf_cb = cb.CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.02,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    rsm=0.2,  # features subsample
    od_type="Iter",  # early stopping odwait = 100, # early stopping
    verbose=100,
    l2_leaf_reg=20,  # regularisation
)


# In[8]:


# clf_cb, dd_cb = sk_model_eval_classification(clf_cb, 1,
#                                               Xtrain, ytrain, Xtest, ytest)


clf_cb.fit(
    Xtrain,
    ytrain,
    eval_set=(Xtest, ytest),
    cat_features=np.arange(0, Xtrain.shape[1]),
    use_best_model=True,
)


# In[ ]:


# In[ ]:


# In[123]:


# Fitting a SVM
clf_svc = SVC(C=1.0, probability=True)  # since we need probabilities

clf_svc, dd_svc = sk_model_eval_classification(clf_svc, 1, Xtrain, ytrain, Xtest, ytest)


# In[228]:


# In[231]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[54]:


clf_nn = MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=80,
    alpha=1e-4,
    activation="relu",
    solver="adam",
    verbose=10,
    tol=1e-4,
    random_state=1,
    learning_rate_init=0.1,
    early_stopping=True,
    validation_fraction=0.2,
)


# In[55]:


clf_nn, dd_nn = sk_model_eval_classification(clf_nn, 1, Xtrain, ytrain, Xtest, ytest)


# # Feature selection
#

# In[ ]:


### Feature Selection (reduce over-fitting)
# Pre model feature selection (sometimes some features are useful even with low variance....)
# Post model feature selection


# In[59]:


### Model independant Selection
colX_kbest = sk_model_eval_feature(
    clf_nn, method="f_classif", colname=colX, kbest=50, Xtrain=Xtrain, ytrain=ytrain
)


print(colX_kbest)


# In[99]:


clf_log_feat


# In[80]:


clf_log.fit(dfX[colX].values, df[coly].values)


# In[100]:


feat_eval = sk_feature_evaluation(
    clf_log, dfX, 30, colname_best=clf_log_feat.feature.values, dfy=df[coly]
)


# In[95]:


feat_eval


# # Ensembling

# In[ ]:


# In[54]:


clf_list = []
clf_list.append(("clf_log", clf_log))
clf_list.append(("clf_lgb", clf_lgb))
clf_list.append(("clf_svc", clf_svc))


clf_ens1 = VotingClassifier(clf_list, voting="soft")  # Soft is required
print(clf_ens1)


# In[55]:


sk_model_eval_classification(clf_ens1, 1, Xtrain, ytrain, Xtest, ytest)


# In[ ]:


# In[ ]:


# In[ ]:


# # Predict values

# In[129]:


dft = pd.read_csv(folder + "/data/address_matching_data.csv")


# In[130]:


#####
dft = dft.replace("?", np.nan)


# In[131]:


dft[colcat].head(3)


# In[132]:


#### Pre-processing  cat :  New Cat are discard, Missing one are included
for col in colcat:
    try:
        dft[col] = dft[col].apply(lambda x: colcat_map["cat_map"][col].get(x))
    except Exception as e:
        print(col, e)


dft_colcat_hot = pd_col_to_onehot(dft[colcat], colcat)


for x in colcat_hot:
    if not x in dft_colcat_hot.columns:
        dft_colcat_hot[x] = 0
        print(x, "added")


dft_colcat_hot[colcat_hot].head(5)


# In[133]:


dft_colcat_hot.head(4)


# In[151]:


#### Pre-processing num :  REUSE Colnum_map and Pad the missing values
dft_numbin, _ = pd_colnum_tocat(
    dft[colnum],
    colname=colnum,
    colexclude=None,
    colbinmap=colnum_map,
    bins=0,
    suffix="_bin",
    method="",
)


# In[157]:


dft_numbin.head(5)


# In[167]:


dft_num_hot = pd_col_to_onehot(dft_numbin[colnum_bin], colname=colnum_bin, colonehot=colnum_hot)


# In[165]:


dft_num_hot.head(5)


# In[168]:


print(dft_num_hot.shape, dfnum_hot.shape)


# In[161]:


# In[169]:


#### Train
X = pd.concat((dft_num_hot, dfcat_hot), axis=1).values

print(X.shape)


# In[175]:


dft[coly] = clf_ens1.predict(X)


# In[ ]:


# In[176]:


dft.head(5)


# In[177]:


dft.groupby(coly).agg({"id": "count"})


# In[172]:


# In[183]:


dft[["id", "is_match"]].to_csv("adress_pred.csv", index=False, mode="w")


# In[ ]:


# In[ ]:


# In[ ]:


# In[174]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[33]:


# In[34]:


### NA handling
df_ = train_df.append(test_df)
use_columns = [s for s in df_.columns if s not in ["id", "is_match"]]
df = df_[use_columns]

for c in df.columns:
    if df[c].dtype == "object":
        df.loc[df[c] == "?", c] = 0
    else:
        print("skip ", c)


# In[35]:


### Encode numerical into Category to handle NA distribution


# In[179]:


0


# In[38]:


# In[39]:


# In[16]:


# In[ ]:


# In[40]:


# In[41]:


# In[ ]:


# In[ ]:


# In[180]:


0


# In[ ]:


"""
Sparse Logistics


"""


# In[181]:


0


# In[46]:


### NO Null Features
len(df_featlogis[df_featlogis["coef_abs"] > 0.0])


# In[ ]:


# In[ ]:


# In[ ]:


preds = clf.predict(test_df[feats])
preds[preds == 0] = -1


# In[ ]:


test_df = pd.read_csv("./data/address_matching_test.csv")
test_df["is_match"] = preds.astype(int)
test_df = test_df.reset_index()
test_df[["id", "is_match"]].to_csv("result.csv", index=False)
