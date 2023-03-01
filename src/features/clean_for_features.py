import pandas as pd
import os

# DOES IT MAKE SENSE TO RE-DO THIS OPTIMIZATION OR DO WE JUST USE THE MODEL WE ALREADY HAVE BY RE-READING DATA?
def optimize_cleaning(data, **params):
	data.loc[:, 'hour'] = data.loc[:, params['time_changed']].transform(lambda x: x.hour)
	medians = data.groupby(params['time_changed']).median()
	hour_medians = data.groupby(['hour']).median()

	min_ts = medians.index[0]
	max_ts = medians.index[len(medians) - 1]

	missingtimes_df = pd.DataFrame(index = pd.date_range(min_ts, max_ts, freq=params['time_floor_val']))

	complete_times = missingtimes_df.merge(medians, left_index = True, right_index = True, how = 'outer')
	complete_times.loc[:, 'hour'] = complete_times.index.hour

	keep_cols = list(complete_times.columns)[0: len(complete_times.columns) - 1]
	keep_cols_y = [x + "_y" for x in keep_cols]

	# merge then keep the relevant columns based on merge logic (no the most efficient but didn't have time to clean)
	imputed_meds = complete_times.loc[complete_times[params['energy_col']].isna(), :].merge(hour_medians, left_on = 'hour', right_index = True)
	imputed_meds = imputed_meds.loc[:, keep_cols_y].rename(dict(zip(keep_cols_y, keep_cols)), axis = 1)

	complete_times.loc[(complete_times[params['energy_col']].isna()), :] = imputed_meds

	return complete_times.reset_index().rename({'index': params['time_col']}, axis = 1).drop(['hour'], axis = 1)

def non_optimize_cleaning(data, **params):
	split_date = params['split_date']

	# split into train and test for impute purposes - train set getting hour column to be used properly
	# based on a date for approximately 70/30% split where 30% is most recent
	train_set = data.loc[data[params['time_changed']] < split_date, :]
	train_set.loc[:, 'hour'] = train_set.loc[:, params['time_changed']].transform(lambda x: x.hour)

	test_set = data.loc[~(data[params['time_changed']] < split_date), :]
	test_set.loc[:, 'train'] = False

	training_percentage = train_set.shape[0] / data.shape[0] * 100
	testing_percentage = test_set.shape[0] / data.shape[0] * 100

	print("*****")
	print('Training percentage of data: {0:.2f}%, testing percentage of data: {1:.2f}%.'.format(training_percentage, testing_percentage))
	print("*****")

	# use medians for each hour as the data, then impute with medians based on hour medians in the original train data
	medians = train_set.groupby(params['time_changed']).median()

	min_ts = medians.index[0]

	# FIX DOCUMENTATION - originally did this with max_ts but split_date makes more sense
	missingtimes_df = pd.DataFrame(index = pd.date_range(min_ts, split_date, freq=params['time_floor_val']))

	complete_times_train = missingtimes_df.merge(medians, left_index = True, right_index = True, how = 'outer')
	complete_times_train.loc[:, 'hour'] = complete_times_train.index.hour

	hour_medians = train_set.groupby(['hour']).median()

	# removing the last column - hour before merging imputed values with the new values
	keep_cols = list(complete_times_train.columns)[0: len(complete_times_train.columns) - 1]
	keep_cols_y = [x + "_y" for x in keep_cols]

	# merge then keep the relevant columns based on merge logic (no the most efficient but didn't have time to clean)
	imputed_meds = complete_times_train.loc[complete_times_train[params['energy_col']].isna(), :].merge(hour_medians, left_on = 'hour', right_index = True)
	imputed_meds = imputed_meds.loc[:, keep_cols_y].rename(dict(zip(keep_cols_y, keep_cols)), axis = 1)

	complete_times_train.loc[(complete_times_train[params['energy_col']].isna()), :] = imputed_meds

	# essentially, handles the case where a certain hour value doesn't have its on median (specifically more common in test case)
	# this is inherently flawed in the test case because it means the model will likely predict to baseline value which is very common
	if complete_times_train[params['energy_col']].isna().shape[0] != 0:
		overall_medians = train_set.median(numeric_only = True).drop(['hour'])

		complete_times_train.loc[(complete_times_train[params['energy_col']].isna()), :] = complete_times_train.loc[(complete_times_train[params['energy_col']].isna()), :].fillna(overall_medians)
		
		# need to run again to handle nulls from previous statement
		complete_times_train.loc[:, 'hour'] = complete_times_train.index.hour

	# making the median test set and defining which variables correspond to a training and testing set
	complete_times_train.loc[:, 'train'] = True

	medians_test = test_set.groupby(params['time_changed']).median()
	medians_test.index = medians_test.index.rename('index')
	medians_test.loc[:, 'train'] = False

	return pd.concat([complete_times_train.reset_index(), medians_test.reset_index()]).drop(['hour'], axis = 1).rename({'index': params['time_col']}, axis = 1)


### MAIN FUNCTION ###
def clean_raw(cwd, data, is_train, is_optimize, **params):
	final_name = params['out_name']

	if is_train:
		if os.path.isdir(cwd + params['temp_output']):
			files = os.listdir(cwd + params['temp_output'])

			if final_name in files:
				print('Timestamped data already found - will use instead of regenerating for time saving.')
				print('If you would like to regenerate this file, please run "python3 run.py clean" in the terminal before calling features again.')

				return pd.read_csv(cwd + params['temp_output'] + final_name)
		else:
			os.mkdir(cwd + params['temp_output'])
	elif not is_optimize:
		print("no run -> data call because test data is already present")
		print("in run -> features")

	# temp time strips UTC addition and then converts to timestamp
	data.loc[:, params['time_col']] = data.loc[:, params['time_col']].str[0:-6].apply(lambda x: pd.Timestamp(x))

	# floor data timestamps to the nearest hour here and get hours
	data.loc[:, params['time_col']] = data.loc[:, params['time_col']].apply(lambda x: x.floor(freq = params['time_floor_val']))

	data = data.rename({params['time_col']: params['time_changed']}, axis = 1)

	# handles training and test cases
	if not is_optimize:
		data = non_optimize_cleaning(data, **params)
	else:
		data = optimize_cleaning(data, **params)


	if is_train or is_optimize:
		data.to_csv(cwd + params['temp_output'] + final_name, index = False)
	else:
		data.to_csv(cwd + params['test_directory'] + final_name, index = False)

	return data

