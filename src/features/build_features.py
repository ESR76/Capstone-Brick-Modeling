import pandas as pd
import os


def time_features(cwd, data, **params):
	print("in features..")
	final_name = params['final_name']

	if os.path.isdir(cwd + params['final_output']):
		files = os.listdir(cwd + params['final_output'])

		if final_name in files:
			print('Timestamped data already found.')
			return pd.read_csv(cwd + params['final_output'] + final_name)
	else:
		os.mkdir(cwd + params['final_output'])

	# creating time column
	data['time_transformed'] = data['time'].apply(lambda x: pd.Timestamp(x))

	# creating other columns
	data['month'] = data['time_transformed'].transform(lambda x: x.month)
	data['year'] = data['time_transformed'].transform(lambda x: x.year)
	data['day'] = data['time_transformed'].transform(lambda x: x.day)
	data['weekday'] = data['time_transformed'].transform(lambda x: x.weekday)
	data['hour'] = data['time_transformed'].transform(lambda x: x.hour)
	data['minute'] = data['time_transformed'].transform(lambda x: x.minute)
	data['second'] = data['time_transformed'].transform(lambda x: x.second)

	data = data.drop(['time_transformed', 'time'], axis = 1)

	data.to_csv(cwd + params['final_output'] + final_name)

	return data