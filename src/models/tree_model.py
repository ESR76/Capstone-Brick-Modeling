from sklearn import tree
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os

# legacy imports from notebooks
#from sklearn import preprocessing
#from sklearn import utils



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

	timestamp_col = params["timestamp_col_tree"]

	dates = data[timestamp_col]

	dates_train = (dates < pd.Timestamp(params["split_date"]))

	training_data = data.loc[dates_train, :].reset_index(drop = True).drop([timestamp_col], axis = 1)
	training_percentage = training_data.shape[0] / data.shape[0] * 100
	testing_data = data.loc[~dates_train, :].reset_index(drop = True).drop([timestamp_col], axis = 1)
	testing_percentage = testing_data.shape[0] / data.shape[0] * 100

	clf = tree.DecisionTreeRegressor(max_depth = params['max_depth'], min_samples_split = params['min_samples_split'])
	
	Xtrain = training_data.drop([energy_col], axis = 1)
	Ytrain = training_data[energy_col]
	
	Xtest = testing_data.drop([energy_col], axis = 1)
	Ytest = testing_data[energy_col]

	clf = clf.fit(Xtrain, Ytrain)
	y_pred = clf.predict(Xtest)
	pred_series = pd.Series(y_pred).rename("preds")
	mse = mean_squared_error(Ytest,pred_series)
	
	print('Training percentage of data: {0:.2f}%, testing percentage of data: {1:.2f}%, prediction mean squared error: {2:.2f}.'.format(training_percentage, testing_percentage, mse))

	if is_train:
		training_data.to_csv(cwd + params['final_output'] + train_name)
		testing_data.to_csv(cwd + params['final_output'] + test_name)

		pred_series.to_csv(cwd + params['final_output'] + final_name)

	return pred_series