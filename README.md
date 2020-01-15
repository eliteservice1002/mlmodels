# mlmodels

```
Lightweight Functional interface to wrap access to Deep Learning, RLearning models
and Hyper Params Search
Logic follows Scikit Learn API and simple for easy extentions logic.
Goal to facilitate Prototype to Prod code.


#### Docs here:   https://mlmodels.readthedocs.io/en/latest/

#### Pypi here :  https://pypi.org/project/mlmodels/

#################################################################################################
Install as editable package   ONLY dev branch

cd yourfolder
git clone https://github.com/arita37/mlmodels.git mlmodels
cd mlmodels
git checkout dev     


pip install -e .  --no-deps  


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







### Test, in CLI type :
ml_models

ml_optim    





#################################################################################################
######### Entry CLI  ############################################################################
ml_models  :  mlmodels/models.py
              Lightweight Functional interface to execute models

ml_optim   :  mlmodels/optim.py
              Lightweight Functional interface to wrap Hyper-parameter Optimization

ml_test    :  A lot of tests




ml_models --do  
    model_list  :  list all models in the repo                            
    testall     :  test all modules inside model_tf
    test        :  test a certain module inside model_tf
    fit         :  wrap fit generic m    ethod
    predict     :  predict  using a pre-trained model and some data
    generate_config  :  generate config file from code source


ml_optim --do
   test      :  Test the hyperparameter optimization for a specific model
   test_all  :  TODO, Test all
   search    :  search for the best hyperparameters of a specific model



##################################################################################################
######### Command line sample (test) #############################################################

#### generate config file
ml_models  --do generate_config  --model_uri model_tf.1_lstm.py  --save_folder "c:\myconfig\"


### TF LSTM model
ml_models  --model_uri model_tf/1_lstm.py  --do test


#### Custom  Models
ml_models --do test  --model_uri "D:\_devs\Python01\gitdev\mlmodels\mlmodels\model_tf\1_lstm.py"


### RL model
ml_models  --model_uri model_tf/rl/4_policygradient  --do test


## PyTorch models
ml_models  --model_uri model_tch/mlp.py  --do test


###### Model param search test
ml_optim --do test


##### #for normal optimization search method
ml_optim --do search --ntrials 1  --config_file optim_config.json --optim_method normal
ml_optim --do search --ntrials 1  --config_file optim_config.json --optim_method prune  ###### for pruning method


###### HyperParam standalone run
ml_optim --modelname model_tf.1_lstm.py  --do test
ml_optim --modelname model_tf.1_lstm.py  --do search


###### Model param search test
python optim.py --do test
python optim.py --do search --ntrials 1  --config_file optim_config.json --optim_method normal
python optim.py --do search --ntrials 1  --config_file optim_config.json --optim_method prune

###### HyperParam standalone run
python optim.py --modelname model_tf.1_lstm.py  --do test
python optim.py --modelname model_tf.1_lstm.py  --do search








####################################################################################################
#########  Interface ###############################################################################
models.py 
   module_load(model_uri)
   model_create(module)
   fit(model, module, session, data_pars, out_pars   )
   metrics(model, module, session, data_pars, out_pars)
   predict(model, module, session, data_pars, out_pars)
   save()
   load()



optim.py
   optim(modelname="model_tf.1_lstm.py",  model_pars= {}, data_pars = {}, compute_pars={"method": "normal/prune"}, save_folder="/mymodel/", log_folder="", ntrials=2) 

   optim_optuna(modelname="model_tf.1_lstm.py", model_pars= {}, data_pars = {}, compute_pars={"method" : "normal/prune"}, save_folder="/mymodel/", log_folder="", ntrials=2) 


### Generic parameters :
   Define in models_config.json
   model_params      :  model definition 
   compute_pars      :  Relative to  the compute
   data_pars         :  Relative to the data
   out_pars          :  Relative to out





#################################################################################################### 
#################################################################################################### 
Models are stored in model_XX/  or in folder XXXXX
    module :  folder/mymodel.py, contains the methods, operations.
    model  :  Class in mymodel.py containing tihe model definition, compilation

*How to define a custom model ?*
   Create a folder,
   Create file mymodel.py

   Include those classes/functions :
      Class Model()                  :   Model definition
            __init__(model_param):
                                  
      def fit(model, compute_pars, )     : train the model
      def predict(model,sess, )         : predic the results
      def get_pars()                    : example of parameters of the model
      def get_dataset(data_pars)        : load dataset
      def test_local()                  : example running the model     
      def test_generic()                : example running the model in global settings  

      def save()                        : save the model
      def load()                        : load the trained model



  Template is available in mlmodels/template/model_XXXX.py
                           model_tf/1_lstm.py





####################################################################################################
######### Code sample  #############################################################################
from mlmodels.models import module_load, data_loader, create_model, fit, predict, stats

model_pars = { "learning_rate": 0.001, "num_layers": 1,
                  "size": ncol_input, "size_layer": 128, "output_size": ncol_output, "timestep": 4,
                  "epoch": 2,}
data_pars = {}

module = models.module_load( model_uri="model_tf.1_lstm.py" )  #Load file definition
model =  models.model_create(module, model_pars)    # Create Model instance
sess =   models.fit(model, module, data_pars)       # fit the model
dict_stats = models.metrics( model, sess, ["loss"])     # get stats

model.save( "myfolder/", model, module, sess,)


model = module.load(folder)    #Create Model instance
module.predict(model, module, data_pars)     # predict pipeline



#df = data_loader(data_pars)

# module, model = module_load_full("model_tf.1_lstm.py", dict_pars= model_pars)  # Net












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
conda install  -c anaconda  tensorflow=1.13.1
conda install -c anaconda scikit-learn pandas matplotlib seaborn -y
conda install -c anaconda  ipykernel spyder-kernels=0.* -y



```











