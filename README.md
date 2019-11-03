# mlmodels

```
#### Docs here:
https://dsa.readthedocs.io/en/latest/



###############################################################################
Install as editable package   ONLY dev branch


cd yourfolder
git clone https://github.com/arita37/mlmodels.git dsa
cd dsa
git checkout dev     

pip install -e .



###### In Jupyter / python Editor   ###########################################
from mlmodels.mlmodels import ztest
ztest.run()









###############################################################################
Naming convention for functions, arguments :

For module file has LIMITED dependency and logic flow :
   util_feature.py  :  Input/Outout should be pandas, pandas-like
   util_model.py :     Input/Output should be numpy, all model evaluation related.
   util_plot.py   :    Input * mostly numpy or pandas
   util_date.py :      for dates related, mostly for pandas processing


#### Dependencies and usage ###########################################
pandas, numpy, scipy, sklearn
from util_feature import *
from util_date import *




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











