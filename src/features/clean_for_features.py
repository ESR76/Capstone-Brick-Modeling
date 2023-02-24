import pandas as pd
import os

def clean_raw(cwd, data, is_train, **params):
	final_name = params['pre_model_name']

	if is_train:
		if os.path.isdir(cwd + params['temp_output']):
			files = os.listdir(cwd + params['temp_output'])

			if final_name in files:
				print('Timestamped data already found - regenerating because of features call.')
		else:
			os.mkdir(cwd + params['temp_output'])
	else:
		print("no run -> data call because test data is already present")
		print("in run -> features")

	data['temp_time'] = data[params['time_col']].str[0:-6].apply(lambda x: pd.Timestamp(x))
	# floor data timestamps to the nearest hour here
	data[params['time_changed']] = data['temp_time'].transform(lambda x: pd.floor(x, params['time_floor_val']))

	data = data.drop([params['time_col'], 'temp_time'], axis = 1)

	if is_train:
		data.to_csv(cwd + params['temp_output'] + final_name)
	else:
		data.to_csv(cwd + params['test_directory'] + final_name)

	return data

