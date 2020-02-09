# Time Series Modeling 

This directory contains differnet models regarding time series models using GluonTS Toolkit.
This gives an end to end api .For now we have created api for FFN , DeepAR and Prophet.<br/> 
Different functionalities api can do 
* get_dataset(data_params)  -- Load dataset
* fit(model, data_pars=None,...) -- Train the model
* save(model, path) -Save model on specified path
* predict(model, data_pars, compute_pars=None,...) -- Predict result 
* metrics(ypred, data_pars, compute_pars=None,...) --Create metrics from output result. 



