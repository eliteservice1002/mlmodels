# mlmodels

```
Lightweight Functional interface to wrap access to Deep Learning, RLearning models.
Logic follows Scikit Learn API and simple for easy extentions logic.
Goal to facilitate Jupyter to Prod.



#### Docs here:
https://mlmodels.readthedocs.io/en/latest/



####################################################################################################
Install as editable package   ONLY dev branch

cd yourfolder
git clone https://github.com/arita37/mlmodels.git mlmodels
cd mlmodels
git checkout dev     

pip install -e .


####################################################################################################
###### In Jupyter / python Editor   ################################################################
from mlmodels.util import load_config, to_namespace
from mlmodels.models import create, module_load, save
from mlmodels import ztest
ztest.run()






####################################################################################################
######### Entry CLI  ###############################################################################
ml_models --do  
    "model_list"  :  #list all models in the repo                            
    "testall"  :                                 
    "test"  :
    "fit"  :
    "predict"  :
    "generate_config"  :


ml_optim --do




####################################################################################################
######### Command line sample (test) ###############################################################
#### generate config file
python mlmodels/models.py  --do generate_config  --model_uri model_tf.1_lstm.py  --save_folder "c:\myconfig\"


#### Custom Directory Models
python mlmodels/models.py --do test  --model_uri "D:\_devs\Python01\gitdev\mlmodels\mlmodels\model_tf\1_lstm.py"


### RL model
python  models.py  --model_uri model_tf/rl/4_policygradient  --do test


### TF LSTM model
python  models.py  --model_uri model_tf/1_lstm.py  --do test


## PyTorch models
python  models.py  --model_uri model_tch/mlp.py  --do test


###### Model param search test
python optim.py --do test


##### #for normal optimization search method
python optim.py --do search --ntrials 1  --config_file optim_config.json --optim_method normal

python optim.py --do search --ntrials 1  --config_file optim_config.json --optim_method prune  ###### for pruning method



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




#################################################################################################### 
Models are stored in model_XX/  or in folder XXXXX
    module :  folder/mymodel.py, contains the methods, operations.
    model  :  Class in mymodel.py containing the model definition, compilation



*How to define a custom model ?*
   Create a folder,
   Create file mymodel.py

   Include those classes/functions :
      Class Model()                  :   Model definition
            __init__(model_param):
                        

      def load_dataset()                  
      def fit(model, )               : train the model
      def predict(model,sess, )      : predic the results
      def get_pars()                 : example of parameters of the model
      def save()                     : save the model
      def load()                     : load the trained model
      def test()                     : example running the model     
      def data_loader(data_pars)







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











