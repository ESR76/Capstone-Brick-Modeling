from sklearn import tree
import pandas as pd
import numpy as np
import os

def reduce_setpoint(x, val, min_val = 0):
	reduced = x - val
	return max(reduced, min_val)

def low_barrier(x, val, min_val = 0):
	reduced = x - val
	if reduced < min_val:
		return True
	return False

def optimize_model(cwd, clf, is_train, **params):
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

			if final_name in files and is_train:
				print('Optimize data already found - will skip regenerating to save time.')
				print('To regenerate - please run "python3 run.py clean" before calling optimize again.')

				return pd.read_csv(cwd + direc + final_name)
	else:
		os.mkdir(cwd + params['optimize_versions_folder'])	

	data = pd.read_csv(cwd + direc + params['train_data'])

	Xtrain = data.drop([output_col], axis = 1)
	cols = Xtrain.columns
	Ytrain = data.loc[:, output_col]

	feature_importances = clf.feature_importances_
	with open(cwd + direc + params['optimization_weights'], 'w') as f:
		f.write('"Feature","Importance"\n')
		for i, importance in enumerate(feature_importances):
			f.write('"' + cols[i] + '", "' + str(importance) + '"\n')

	opt_options = params["optimize_options"]
	columns = list(opt_options.keys())

	overall = []
	dfs = []
	results = []

	Xtest = Xtrain.copy(deep = True)
	max_temp_change = opt_options[columns[0]][-1]
	max_temp_vals = []
	max_air_change = opt_options[columns[1]][-1]
	max_air_vals = []

	# Makes 3 versions, one for high occupancy minimum, one for low occupancy min, one for no occupancy (0)
	for a in opt_options[columns[1]]:
		for t in opt_options[columns[0]]:
			# occupied - low limit
			Xtest.loc[:, columns[0]] = Xtrain.loc[:, columns[0]].apply(reduce_setpoint, args = (t, ))
			Xtest.loc[:, columns[1]] = Xtrain.loc[:, columns[1]].apply(reduce_setpoint, args = (a, params['optimization_room_min'], ))
			limited = Xtrain.loc[:, columns[1]].apply(low_barrier, args = (a, params['optimization_room_avgmin'], ))

			y_pred = clf.predict(Xtest)
			pred_series = pd.Series(y_pred).rename("preds")
			differences = Ytrain - pred_series

			if max_temp_change == t:
				max_temp_vals.extend(differences)
			if max_air_change == a:
				max_air_vals.extend(differences)

			groups = pd.DataFrame({params['optimization_group_col']: Xtest[params['optimization_group_col']], 'differences': differences, 'was_limited': limited})

			reduced_groups = groups.groupby(params['optimization_group_col'])['differences'].agg(['sum', 'min', 'max', 'mean', 'median'])
			limited_sums = groups.groupby(params['optimization_group_col'])['was_limited'].sum().rename({'sum':'prop'})
			limited_counts = groups.groupby(params['optimization_group_col'])['was_limited'].count().fillna(0)
			props = limited_sums / limited_counts

			final_groups = reduced_groups.merge(props, left_index = True, right_index = True)

			final_groups.loc[:, 'temp_set'] = t
			final_groups.loc[:, 'air_set'] = a
			final_groups.loc[:, 'occupancy'] = 'low'

			dfs.append((groups, t, a, 'low'))
			results.append(final_groups.reset_index())
			overall.append(groups)



			# occupied - high limit
			Xtest.loc[:, columns[1]] = Xtrain.loc[:, columns[1]].apply(reduce_setpoint, args = (a, params['optimization_room_avgmin'], ))
			limited = Xtrain.loc[:, columns[1]].apply(low_barrier, args = (a, params['optimization_room_avgmin'], ))

			y_pred = clf.predict(Xtest)
			pred_series = pd.Series(y_pred).rename("preds")
			differences = Ytrain - pred_series

			groups = pd.DataFrame({params['optimization_group_col']: Xtest[params['optimization_group_col']], 'differences': differences, 'was_limited': limited})

			reduced_groups = groups.groupby(params['optimization_group_col'])['differences'].agg(['sum', 'min', 'max', 'mean', 'median'])
			limited_sums = groups.groupby(params['optimization_group_col'])['was_limited'].sum().rename({'sum':'prop'})
			limited_counts = groups.groupby(params['optimization_group_col'])['was_limited'].count().fillna(0)
			props = limited_sums / limited_counts

			final_groups = reduced_groups.merge(props, left_index = True, right_index = True)

			final_groups.loc[:, 'temp_set'] = t
			final_groups.loc[:, 'air_set'] = a
			final_groups.loc[:, 'occupancy'] = 'high'

			dfs.append((groups, t, a, 'high'))
			results.append(final_groups.reset_index())
			overall.append(groups)

			if max_temp_change == t:
				max_temp_vals.extend(differences)
			if max_air_change == a:
				max_air_vals.extend(differences)

			# unoccupied
			Xtest.loc[:, columns[1]] = Xtrain.loc[:, columns[1]].apply(reduce_setpoint, args = (a, ))
			limited = Xtrain.loc[:, columns[1]].apply(low_barrier, args = (a, ))

			y_pred = clf.predict(Xtest)
			pred_series = pd.Series(y_pred).rename("preds")
			differences = Ytrain - pred_series

			groups = pd.DataFrame({params['optimization_group_col']: Xtest[params['optimization_group_col']], 'differences': differences, 'was_limited': limited})

			reduced_groups = groups.groupby(params['optimization_group_col'])['differences'].agg(['sum', 'min', 'max', 'mean', 'median'])
			limited_sums = groups.groupby(params['optimization_group_col'])['was_limited'].sum().rename({'sum':'prop'})
			limited_counts = groups.groupby(params['optimization_group_col'])['was_limited'].count().fillna(0)
			props = limited_sums / limited_counts

			final_groups = reduced_groups.merge(props, left_index = True, right_index = True)

			final_groups.loc[:, 'temp_set'] = t
			final_groups.loc[:, 'air_set'] = a
			final_groups.loc[:, 'occupancy'] = 'unoccupied'

			dfs.append((groups, t, a, 'unoccupied'))
			results.append(final_groups.reset_index())
			overall.append(groups)

			if max_temp_change == t:
				max_temp_vals.extend(differences)
			if max_air_change == a:
				max_air_vals.extend(differences)
	
	print("*****")
	print('Median Cost Change at Temp Reduction = {0} degrees F: {1:.4f}.'.format(max_temp_change, np.median(max_temp_vals)))
	print('Median Cost Change at Airflow Reduction = {0} CFM: {1:.4f}.'.format(max_air_change, np.median(max_air_vals)))
	print("*****")

	pred_df = pd.concat(results)
	total_df = pd.concat(overall)

	# doing the total aggreagates for the whole
	reduced_groups = total_df.groupby(params['optimization_group_col'])['differences'].agg(['sum', 'min', 'max', 'mean', 'median'])
	limited_sums = total_df.groupby(params['optimization_group_col'])['was_limited'].sum().rename({'sum':'prop'})
	limited_counts = total_df.groupby(params['optimization_group_col'])['was_limited'].count().fillna(0)
	props = limited_sums / limited_counts

	final_groups = reduced_groups.merge(props, left_index = True, right_index = True).reset_index()

	for df in dfs:
		df[0].to_csv(cwd + params['optimize_versions_folder'] + 'optimize_t{0}_a{1}_{2}.csv'.format(str(df[1]).replace(".", ""), df[2], df[3]), index = False)

	if is_train:
		pred_df.to_csv(cwd + params['final_output'] + final_name, index = False)
		final_groups.to_csv(cwd + params['final_output'] + 'total_' + final_name, index = False)
	else:
		pred_df.to_csv(cwd + params['test_directory'] + final_name, index = False)
		final_groups.to_csv(cwd + params['test_directory'] + 'total_' + final_name, index = False)
	
	return pred_df