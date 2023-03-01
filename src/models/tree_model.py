from sklearn import tree
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os

def regular_model(data, **params):
	output_col = params['output_col']

	training_data = data.loc[data.loc[:, "train"], :].reset_index(drop = True).drop(["train"], axis = 1)
	testing_data = data.loc[~data.loc[:, "train"], :].reset_index(drop = True).drop(["train"], axis = 1)

	clf = tree.DecisionTreeRegressor(max_depth = params['max_depth'], min_samples_split = params['min_samples_split'])
	
	Xtrain = training_data.drop([output_col], axis = 1)
	Ytrain = training_data[output_col]

	Xtest = testing_data.drop([output_col], axis = 1)
	Ytest = testing_data[output_col]

	clf = clf.fit(Xtrain, Ytrain)
	y_pred = clf.predict(Xtest)
	pred_series = pd.Series(y_pred).rename("preds")

	mse = mean_squared_error(Ytest,pred_series)

	# needed to move from 2 decimal points to 4 because it looks like it's doing really well on the cost data?
	# like even better than the other one?
	print("*****")
	print('MSE: {0:.4f}.'.format(mse))
	print("*****")

	return training_data, testing_data, pred_series

def optimize_model(data, **params):
	if os.path.isdir(cwd + params['optimize_versions_folder']):
		files = os.listdir(cwd + params['optimize_versions_folder'])

		if len(files) != 0:
			print('Optimize test set folder already populated - regenerating because of call to re-optimize.')
	else:
		os.mkdir(cwd + params['optimize_versions_folder'])

	testing_datasets

	# NEED TO FILL THIS IN
	# want to create several different versions of the dataset and then run them each through the model


def generate_model(cwd, data, is_train, is_optimize, **params):
	print("in model..")

	train_name = params['train_data']
	test_name = params['test_data']
	final_name = params['modeled_preds']

	if is_train:
		if os.path.isdir(cwd + params['final_output']):
			files = os.listdir(cwd + params['final_output'])

			if final_name in files:
				print('Modeled data already found - regenerating because of model call.')
		else:
			os.mkdir(cwd + params['final_output'])
	elif not is_optimize:
		print("in run -> model")

	if not is_optimize:
		training_data, testing_data, pred_series = regular_model(data, **params)
	else:
		dfs, pred_series = optimize_model(data, **params)
	
	if is_train and not is_optimize:
		training_data.to_csv(cwd + params['final_output'] + train_name, index = False)
		testing_data.to_csv(cwd + params['final_output'] + test_name, index = False)

		pred_series.to_csv(cwd + params['final_output'] + final_name, index = False)
	elif not is_optimize:
		training_data.to_csv(cwd + params['test_directory'] + train_name, index = False)
		testing_data.to_csv(cwd + params['test_directory'] + test_name, index = False)

		pred_series.to_csv(cwd + params['test_directory'] + final_name, index = False)
	else:
		print("not done yet")

	return pred_series