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
	data['minute'] = data[time_col].transform(lambda x: x.minute)
	data['second'] = data[time_col].transform(lambda x: x.second)

	data = data.drop([time_col], axis = 1)

	return data

# unused currently - features required for the prophet version
def create_prophet_features(data, predictor, output, **params):
	data_subset = data.loc[:, [predictor, output]]

	data_subset = data_subset.rename({predictor: 'ds', output: 'y'}, axis = 1)

	data_subset[params['time_changed']] = data_subset['ds'].transform(pd.Timestamp)

	return data_subset

def cost_mod_energy(data, **params):
	data_subset = data.loc[:, [params['time_col'], params['energy_col']]]
	years = data.loc[:, params['time_col']].transform(lambda x: x.year)

	# can't get this comparison to work - fix in morning
	print(type(years))
	print(years[0:3])

	fiscal_values = params['fiscal_values']
	dates = list(fiscal_values.keys())

	compare_ts = pd.Timestamp(dates[1])
	print(years[0])
	print(type(years[0]))
	before_change = years.transform(lambda x: x <= compare_ts)

	data.loc[before_change, params['cost_col']] = data.loc[:, params['energy_col']] * fiscal_values[dates[0]]
	data.loc[~before_change, params['cost_col']] = data.loc[:, params['energy_col']] * fiscal_values[dates[1]]

	return data



def time_features(cwd, data, is_train, **params):
	final_name = params['pre_model_name']

	# creating time column for standard cleaning pipeline
	data.loc[:, params['time_col']] = data.loc[:, params['time_col']].apply(lambda x: pd.Timestamp(x))

	# creates cost column using energy col
	data = cost_mod_energy(data, **params)

	data = create_time_cols(data, params['time_col'])
	# alternate prophet pipeline
	#data = create_prophet_features(data, params['time_col'], params['energy_col'], params)

	data = data.drop([params['time_col'], params['energy_col']], axis = 1)

	if is_train:
		data.to_csv(cwd + params['temp_output'] + final_name, index = False)
	else:
		data.to_csv(cwd + params['test_directory'] + final_name)

	return data

