from sklearn import tree
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os

def regular_model(cwd, data, is_train, **params):
	train_name = params['train_data']
	final_name = params['modeled_preds']
	test_name = params['test_data']
	output_col = params['output_col']

	if is_train:
		print("in model..")
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

	if is_train:
		training_data.to_csv(cwd + params['final_output'] + train_name, index = False)
		testing_data.to_csv(cwd + params['final_output'] + test_name, index = False)

		pred_series.to_csv(cwd + params['final_output'] + final_name, index = False)
	else:
		training_data.to_csv(cwd + params['test_directory'] + train_name, index = False)
		testing_data.to_csv(cwd + params['test_directory'] + test_name, index = False)

		pred_series.to_csv(cwd + params['test_directory'] + final_name, index = False)

	return pred_series

def reduce_setpoint(x, val):
	reduced = x - val
	return max(reduced, 0)

def optimize_model(cwd, data, **params):
	if os.path.isdir(cwd + params['optimize_versions_folder']):
		files = os.listdir(cwd + params['optimize_versions_folder'])

		if len(files) != 0:
			print('Optimize test set folder already populated - regenerating because of call to re-optimize.')
	else:
		os.mkdir(cwd + params['optimize_versions_folder'])

	output_col = params['output_col']

	Xtrain = data.drop([output_col], axis = 1)
	Ytrain = data[output_col]

	clf = tree.DecisionTreeRegressor(max_depth = params['max_depth'], min_samples_split = params['min_samples_split'])
	clf = clf.fit(Xtrain, Ytrain)

	print(clf.feature_importances_)

	opt_options = params["optimize_options"]
	columns = list(opt_options.keys())

	dfs = []
	results = []

	Xtest = Xtrain.copy(deep = True)

	# does it make sense to create this as a loop like this
	for t in opt_options[columns[0]]:
		for a in opt_options[columns[1]]:
			Xtest.loc[:, columns[0]] = Xtrain.loc[:, columns[0]].apply(reduce_setpoint, args = (t, ))
			Xtest.loc[:, columns[1]] = Xtrain.loc[:, columns[1]].apply(reduce_setpoint, args = (a, ))

			#print(Xtest.head(2))
			y_pred = clf.predict(Xtest)
			pred_series = pd.Series(y_pred).rename("preds")
			differences = Ytrain - pred_series

			dfs.append(Xtest)
			results.append((t, a, differences.mean(), differences.median(), differences.min(), differences.max()))

	pred_series = pd.DataFrame(results, columns = ['temp_decrease', 'air_decrease', 'mean_difference', 'median_difference', 'min_difference', 'max_difference'])

	if not os.path.isdir(cwd + params['optimize_versions_folder']):
		os.mkdir(cwd + params['optimize_versions_folder'])
	#else:
	#	files = os.listdir(cwd + params['optimize_versions_folder'])

	#	for file in files:
	#		os.remove(file)

	for i, df in enumerate(dfs):
		df.to_csv(cwd + params['optimize_versions_folder'] + 'optimize_t{0}_a{1}.csv'.format(str(pred_series.loc[i, 'temp_decrease']).replace(".", ""), pred_series.loc[i, 'air_decrease']), index = False)

	pred_series.to_csv(cwd + params['final_output'] + 'optimize_results.csv', index = False)

	return pred_series

def generate_model(cwd, data, is_train, is_optimize, **params):
	if not is_optimize:
		pred_series = regular_model(cwd, data, is_train, **params)
	else:
		pred_series = optimize_model(cwd, data, **params)

	return pred_series