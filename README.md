# mlmodels

```
#### Docs here:
https://mlmodels.readthedocs.io/en/latest/



###############################################################################
Install as editable package   ONLY dev branch


cd yourfolder
git clone https://github.com/arita37/mlmodels.git mlmodels
cd mlmodels
git checkout dev     

pip install -e .



###### In Jupyter / python Editor   ###########################################
from mlmodels.util import load_config, to_namespace
from mlmodels.models import create, module_load, save

from mlmodels import ztest
ztest.run()



##Run test model    ##########################################################
python models.py --do test

python optim.py --do test


###### CLI for pruning method
python optim.py --do search --ntrials 1  --config_file optim_config.json --optim_method prune







#### Usage :   ###############################################################
python models.py  --do test

python optim.py  --do test




### Install :

 pip install -e git
 
 
 import mlmodel
 mlmodel.test()
 


*How to define a model ?*

   create file mymodel.py
      Class Model()
            __init__(model_param):
                        
      def fit(model, )       :  train the model
      def predict(model,sess, )  : predic the results
      def get_params() : example of parameters of the model
      def save()   : save the model
      def load()   : load the trained model
      def test()   : example running the model
     
      def data_loader(data_params)








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











