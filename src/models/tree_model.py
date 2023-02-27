from sklearn import tree
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os

def generate_model(cwd, data, is_train, **params):
	print("in model..")

	train_name = params['train_data']
	test_name = params['test_data']
	final_name = params['modeled_preds']
	energy_col = params['energy_col']

	if is_train:
		if os.path.isdir(cwd + params['final_output']):
			files = os.listdir(cwd + params['final_output'])

			if final_name in files:
				print('Modeled data already found - regenerating because of model call.')
		else:
			os.mkdir(cwd + params['final_output'])
	else:
		print("in run -> model")

	training_data = data.loc[data.loc[:, "train"], :].reset_index(drop = True).drop(["train"], axis = 1)
	testing_data = data.loc[~data.loc[:, "train"], :].reset_index(drop = True).drop(["train"], axis = 1)

	clf = tree.DecisionTreeRegressor(max_depth = params['max_depth'], min_samples_split = params['min_samples_split'])
	
	Xtrain = training_data.drop([energy_col], axis = 1)
	Ytrain = training_data[energy_col]

	Xtest = testing_data.drop([energy_col], axis = 1)
	Ytest = testing_data[energy_col]

	clf = clf.fit(Xtrain, Ytrain)
	y_pred = clf.predict(Xtest)
	pred_series = pd.Series(y_pred).rename("preds")
	mse = mean_squared_error(Ytest,pred_series)

	print("*****")
	print('MSE: {0:.2f}.'.format(mse))
	print("*****")
	
	if is_train:
		training_data.to_csv(cwd + params['final_output'] + train_name, index = False)
		testing_data.to_csv(cwd + params['final_output'] + test_name, index = False)

		pred_series.to_csv(cwd + params['final_output'] + final_name, index = False)
	else:
		training_data.to_csv(cwd + params['test_directory'] + train_name)
		testing_data.to_csv(cwd + params['test_directory'] + test_name)

		pred_series.to_csv(cwd + params['test_directory'] + final_name)

	return pred_series