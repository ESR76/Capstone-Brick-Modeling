from sklearn import tree
import pandas as pd
import numpy as np
import os

def reduce_setpoint(x, val):
	reduced = x - val
	return max(reduced, 0)

def optimize_model(cwd, model, is_train, **params):
	final_name = params['optimize_results']
	output_col = params['output_col']
	data = pd.DataFrame()
	direc = ""

	if is_train:
		direc = params['final_output']
	else:
		print("\nin run -> optimize for test data")
		direc = params['test_directory']

	if os.path.isdir(cwd + params['optimize_versions_folder']):
			files = os.listdir(cwd + direc)

			if final_name in files:
				print('Optimize data already found - will skip regenerating to save time.')
				print('To regenerate - please run "python3 run.py clean" before calling optimize again.')

				return pd.read_csv(cwd + direc + final_name)
	else:
		os.mkdir(cwd + params['optimize_versions_folder'])

	data = pd.read_csv(cwd + direc + params['train_data'])

	Xtrain = data.drop([output_col], axis = 1)
	Ytrain = data.loc[:, output_col]

	clf = model
	clf = clf.fit(Xtrain, Ytrain)

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

	# WOULD IT BE POSSIBLE TO USE .DESCRIBE instead of PULLING OUT MAX

	pred_df = pd.DataFrame(results, columns = ['temp_decrease', 'air_decrease', 'mean_difference', 'median_difference', 'min_difference', 'max_difference'])

	for i, df in enumerate(dfs):
		df.to_csv(cwd + params['optimize_versions_folder'] + 'optimize_t{0}_a{1}.csv'.format(str(pred_df.loc[i, 'temp_decrease']).replace(".", ""), pred_df.loc[i, 'air_decrease']), index = False)

	if is_train:
		pred_df.to_csv(cwd + params['final_output'] + final_name, index = False)
	else:
		pred_df.to_csv(cwd + params['test_directory'] + final_name, index = False)
	
	return pred_df