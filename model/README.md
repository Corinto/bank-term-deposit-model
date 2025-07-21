# bank-term-deposit-model
A DS model to predict likelihood of customers taking on term deposits

We have implemented a script to generate a model (binary classifier)
to the dataset provided. 

To run the scripts use this order:
 1.- python model_train.py

 	Trains the classifier on the dataset provided: /bank/bank-full.csv
	Saves the model in folder /saved_models/
	Generates evaluation of model on training set using k-fold validation
 	and hold-out test set.

 	Outputs:
		/saved_models/model.joblib
		/saved_models/model_auc.joblib

 2.- python model_inference.py

	Performs inference using saved model on dataset 
	provided for test. /bank/bank.csv
 	Predictios of the model are saved in folder /predictions
	Two columns are added for model prediction and probablity of
 	the positive class.

	Outputs:
 		/predictions/bank_model_predictions.csv

 3.- python model_inference_eval.py

	Computes accuracy metrics for the predictions on the model 
	for inference dataset.
	
	Outputs:
		Acc metric to stdout
		Displays AUROC curve plot

 All scripts provided can run on new datasets if necessary.
 Please add /saved_models and /predictions folders if needed 
 in your local system to run the scripts.

These scripts have been tested on Macintosh and python V.3.11.5