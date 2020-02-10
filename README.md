# mlmodels

```
Lightweight Functional interface to wrap access to Deep Learning, ML models
and Hyper Params Search.
Logic follows Scikit Learn API for easy extentions logic.
Goal to facilitate Prototype to Semi-Prod code.


#### Docs here:   https://mlmodels.readthedocs.io/en/latest/  (incomplete docs)
#### Pypi here :  https://pypi.org/project/mlmodels/

#################################################################################################
Install as editable package   ONLY dev branch

cd yourfolder
git clone https://github.com/arita37/mlmodels.git mlmodels
cd mlmodels
git checkout dev     
pip install -e .  --no-deps  


### Test, in CLI type :
ml_models
ml_optim    


####  dependencies
numpy>=1.16.4
pandas>=0.24.2
scipy>=1.3.0
scikit-learn>=0.21.2
numexpr>=2.6.8 
sqlalchemy>=1.3.8
tensorflow>=1.14.0
pytorch>=0.4.0
optuna


#################################################################################################
######### Entry CLI  ############################################################################
ml_models  :  mlmodels/models.py
              Lightweight Functional interface to execute models

ml_models --do  
    model_list  :  list all models in the repo                            
    testall     :  test all modules inside model_tf
    test        :  test a certain module inside model_tf
    fit         :  wrap fit generic m    ethod
    predict     :  predict  using a pre-trained model and some data
    generate_config  :  generate config file from code source


ml_optim   :  mlmodels/optim.py
              Lightweight Functional interface to wrap Hyper-parameter Optimization
              
ml_optim --do
   test      :  Test the hyperparameter optimization for a specific model
   test_all  :  TODO, Test all
   search    :  search for the best hyperparameters of a specific model

ml_test      :  A lot of tests


##################################################################################################
######### Command line sample        #############################################################

#### generate config file
ml_models  --do generate_config  --model_uri model_tf.1_lstm.py  --save_folder "c:\myconfig\"


#### TF LSTM model
ml_models  --model_uri model_tf/1_lstm.py  --do test


#### Custom  Models
ml_models --do test  --model_uri "D:\_devs\Python01\gitdev\mlmodels\mlmodels\model_tf\1_lstm.py"


#### PyTorch models
ml_models  --model_uri model_tch/mlp.py  --do test


#### Model param search test
ml_optim --do test


#### For normal optimization search method
ml_optim --do search --ntrials 1  --config_file optim_config.json --optim_method normal
ml_optim --do search --ntrials 1  --config_file optim_config.json --optim_method prune  ###### for pruning method


#### HyperParam standalone run
ml_optim --modelname model_tf.1_lstm.py  --do test
ml_optim --modelname model_tf.1_lstm.py  --do search




#################################################################################################### 
########## How to add a new model ################################################################## 
*How to define a custom model ?*
   Create a folder,
   Create a file mymodel.py

   Include those classes/functions :
      Class Model()                  :   Model definition
            __init__(model_param):
                                  
      def fit(model, data_pars, model_pars, compute_pars, )     : Train the model
      def predict(model, sess, data_pars, compute_pars, out_pars )         : Predict the results
      def metric(ypred, data_pars, compute_pars, out_pars )         : Measure the results

      def get_pars()                    : example of parameters of the model
      def get_dataset(data_pars)        : load dataset
      def test()                        : example running the model     
      def test2()                       : example running the model in global settings  

      def save()                        : save the model
      def load()                        : load the trained model


  Template is available in mlmodels/template/model_XXXX.py



####################################################################################################
#########  Interface ###############################################################################
models.py 
   module_load(model_uri)
   model_create(module)
   fit(model, module, session, data_pars, out_pars   )
   metrics(model, module, session, data_pars, out_pars)
   predict(model, module, session, data_pars, out_pars)
   save(model, path)
   load(model)


optim.py
   optim(modelname="model_tf.1_lstm.py",  model_pars= {}, data_pars = {}, compute_pars={"method": "normal/prune"}
       , save_folder="/mymodel/", log_folder="", ntrials=2) 

   optim_optuna(modelname="model_tf.1_lstm.py", model_pars= {}, data_pars = {}, compute_pars={"method" : "normal/prune"},
                save_folder="/mymodel/", log_folder="", ntrials=2) 


### Generic parameters :
   Define in models_config.json
   model_params      :  Relative to model definition 
   compute_pars      :  Relative to  the compute process
   data_pars         :  Relative to the input data
   out_pars          :  Relative to outout data

   Sometimes, data_pars is required to setup the model (ie CNN with image size...)
   



####################################################################################################
######### Code sample  #############################################################################
from mlmodels.models import module_load, data_loader, create_model, fit, predict, stats
from mlmodels.models import load #Load model weights

#### Training
model_pars   =  {  "num_layers": 1,
                  "size": ncol_input, "size_layer": 128, "output_size": ncol_output, "timestep": 4,
                }
data_pars    =  {}
compute_pars =  { "learning_rate": 0.001, }

module        =  module_load( model_uri="model_tf.1_lstm.py" )  #Load file definition
model         =  model_create(module, model_pars)    # Create Model instance
model, sess   =  fit(model, module, data_pars)       # fit the model
metrics_val   =  metrics( model, sess, ["loss"])     # get stats
model.save( "myfolder/", model, module, sess,)


#### Inference
model = load(folder)    #Create Model instance
ypred = module.predict(model, module, data_pars, compute_pars)     # predict pipeline






###############################################################################
###############################################################################
Naming convention for functions, arguments :

## Function naming   ##################################################
pd_   :  input is pandas dataframe
np_   :  input is numpy
sk_   :  inout is related to sklearn (ie sklearn model), input is numpy array
plot_


_col_  :  name for colums
_colcat_  :  name for category columns
_colnum_  :  name for numerical columns (folat)
_coltext_  : name for text data
_colid_  : for unique ID columns\

_stat_ : show statistics
_df_  : dataframe
_num_ : statistics

col_ :  function name for column list related.



### Argument Variables naming  ###############################################
df     :  variable name for dataframe
colname  : for list of columns
colexclude
colcat : For category column
colnum :  For numerical columns
coldate : for date columns
coltext : for raw text columns



#########
Auto formatting
isort -rc .
black --line-length 100




###### Auto Doc
https://pypi.org/project/publish-sphinx-docs/


#### Doc creations
https://readthedocs.org/projects/dsa/builds/9658212/



#########Conda install    ##################################################
conda create -n py36_tf13 python=3.6.5  -y
source activate py36_tf13


pip install tensorflow=1.13.1
pip install  ipykernel spyder-kernels=0.* -y
conda install  -c anaconda  tensorflow=1.13.1
conda install -c anaconda scikit-learn pandas matplotlib seaborn -y
conda install -c anaconda  ipykernel spyder-kernels=0.* -y



```











