import pandas as pd
import os

def create_time_cols(data, time_col):
	# assumes a timestamp columns has already been created
	data['month'] = data[time_col].transform(lambda x: x.month)
	data['year'] = data[time_col].transform(lambda x: x.year)
	data['day'] = data[time_col].transform(lambda x: x.day)
	data['weekday'] = data[time_col].transform(lambda x: x.weekday)
	data['hour'] = data[time_col].transform(lambda x: x.hour)
	data['minute'] = data[time_col].transform(lambda x: x.minute)
	data['second'] = data[time_col].transform(lambda x: x.second)

	return data


def time_features(cwd, data, is_train, **params):
	print("in features..")

	final_name = params['final_name']

	if is_train:
		if os.path.isdir(cwd + params['final_output']):
			files = os.listdir(cwd + params['final_output'])

			if final_name in files:
				print('Timestamped data already found - regenerating because of features call.')
		else:
			os.mkdir(cwd + params['final_output'])

	# creating time column
	data['time_transformed'] = data[params['time_col']].apply(lambda x: pd.Timestamp(x))

	# creating other columns
	data = create_time_cols(data, 'time_transformed')
	data = data.drop(['time_transformed', 'time'], axis = 1)

	if is_train:
		data.to_csv(cwd + params['final_output'] + final_name)
	else:
		data.to_csv(cwd + params['test_directory'] + final_name)

	return data

