import pandas as pd
import os

def create_time_cols(data, time_col):
	# assumes a timestamp columns has already been created
	data['month'] = data[time_col].transform(lambda x: x.month)
	data['year'] = data[time_col].transform(lambda x: x.year)
	data['day'] = data[time_col].transform(lambda x: x.day)
	data['weekday'] = data[time_col].transform(lambda x: x.weekday())
	data['hour'] = data[time_col].transform(lambda x: x.hour)
	data['minute'] = data[time_col].transform(lambda x: x.minute)
	data['second'] = data[time_col].transform(lambda x: x.second)

	return data

# unused for checkpoint
def create_prophet_features(data, time, energy):
	data_subset = data.loc[:, [time, energy]]

	data_subset[time] = data_subset[time].str[0: -6]

	data_subset = data_subset.rename({time: 'ds', energy: 'y'}, axis = 1)

	data_subset['time_transformed'] = data_subset['ds'].transform(pd.Timestamp)

	return data_subset


def time_features(cwd, data, is_train, **params):
	final_name = params['pre_model_name']

	# no longer need this secion in time_features because it's in clean
	#if is_train:
	#	if os.path.isdir(cwd + params['temp_output']):
	#		files = os.listdir(cwd + params['temp_output'])

	#		if final_name in files:
	#			print('Timestamped data already found - regenerating because of features call.')
	#	else:
	#		os.mkdir(cwd + params['temp_output'])
	#else:
	#	print("no run -> data call because test data is already present")
	#	print("in run -> features")


	# creating time column for standard cleaning pipeline
	data.loc[:, params['time_col']] = data.loc[:, params['time_col']].apply(lambda x: pd.Timestamp(x))
	data = create_time_cols(data, params['time_col'])

	# CREATE COST COLUMN INSTEAD OF ENERGY COLUMN HERE
	# ADD MODIFICATIONS FOR SLIDING SCALE OF COST
	# AND DROP ENERGY WHEN DONE WITH TIME

	# alternate prophet pipeline
	#data = create_prophet_features(data, params['time_col'], params['energy_col'])

	data = data.drop([params['time_col']], axis = 1)

	if is_train:
		data.to_csv(cwd + params['temp_output'] + final_name, index = False)
	else:
		data.to_csv(cwd + params['test_directory'] + final_name)

	return data

