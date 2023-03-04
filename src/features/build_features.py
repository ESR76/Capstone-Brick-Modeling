import pandas as pd
import os

# for tree version of the pipeline
def create_time_cols(data, time_col):
	# assumes a timestamp columns has already been created
	data['month'] = data[time_col].transform(lambda x: x.month)
	data['year'] = data[time_col].transform(lambda x: x.year)
	data['day'] = data[time_col].transform(lambda x: x.day)
	data['weekday'] = data[time_col].transform(lambda x: x.weekday())
	data['hour'] = data[time_col].transform(lambda x: x.hour)

	# Currently commenting these out because there is no difference at the 1 hour flooring
	#data['minute'] = data[time_col].transform(lambda x: x.minute)
	#data['second'] = data[time_col].transform(lambda x: x.second)

	data = data.drop([time_col], axis = 1)

	return data

# unused currently - features required for the prophet version
def create_prophet_features(data, predictor, output, **params):
	data_subset = data.loc[:, [predictor, output]]

	data_subset = data_subset.rename({predictor: 'ds', output: 'y'}, axis = 1)

	data_subset[params['time_changed']] = data_subset['ds'].transform(pd.Timestamp)

	return data_subset

def cost_mod_energy(data, **params):
	fiscal_values = params['fiscal_values']
	dates = list(fiscal_values.keys())

	compare_ts = pd.Timestamp(dates[1])

	before_change = data[params['time_col']].apply(lambda x: x <= compare_ts)

	data.loc[before_change, params['cost_col']] = data.loc[:, params['energy_col']] * fiscal_values[dates[0]]
	data.loc[~before_change, params['cost_col']] = data.loc[:, params['energy_col']] * fiscal_values[dates[1]]

	data = data.drop([params['energy_col']], axis = 1)

	return data

def time_features(cwd, data, is_train, **params):
	final_name = params['pre_model_name']
	direc = ""
	if is_train:
		direc = params['temp_output']
	else:
		direc = params['test_directory']
		print("\nin run -> features pt. 2 for test data")

	files = os.listdir(cwd + direc)
	if final_name in files:
		print("Feature-generated data already found, skipping regeneration.")
		return pd.read_csv(cwd + direc + final_name)

	# creates cost column using energy col
	data = cost_mod_energy(data, **params)

	data = create_time_cols(data, params['time_col'])

	# alternate prophet pipeline
	#data = create_prophet_features(data, params['time_col'], params['cost_col'], params)

	if is_train:
		data.to_csv(cwd + params['temp_output'] + final_name, index = False)
	else:
		data.to_csv(cwd + params['test_directory'] + final_name, index = False)

	return data

