import prophet as Prophet
import pandas as pd
import numpy as np

# This does not fit current logic/params for the pipeline - remnant of an older version

def generate_model(cwd, data, is_train, **params):
	print("in model..")
	timestamp_col = params["timestamp_col_prophet"]

	dates = data[timestamp_col]

	dates_train = (dates < pd.Timestamp(params["split_date"]))

	training_data = data.loc[dates_train, :].reset_index(drop = True).drop([timestamp_col], axis = 1)
	training_percentage = training_data.shape[0] / data.shape[0] * 100
	testing_data = data.loc[~dates_train, :].reset_index(drop = True)
	testing_percentage = testing_data.shape[0] / data.shape[0] * 100

	model = Prophet.Prophet()
	model.fit(training_data)
	prophet_forecast = model.make_future_dataframe(periods = 51000, freq = '5min')
	prophet_forecast = model.predict(prophet_forecast)

	prophet_forecast_reduced = prophet_forecast.loc[:, ['ds', 'yhat']]
	prophet_forecast_reduced[timestamp_col] = prophet_forecast_reduced['ds'].transform(lambda x: pd.Timestamp(x))

	merge_compare = prophet_forecast_reduced.merge(testing_data, left_on = timestamp_col, right_on = timestamp_col)

	mse = ((merge_compare['yhat'] - merge_compare['y']) ** 2).sum() / merge_compare.shape[0]

	print('Training percentage: ' + str(training_percentage) + ', testing percentage: ' + str(testing_percentage) + ', prediction mean squared error: ' + str(mse))
	
	# not sure what I'm doing wrong with this line right now but leaving it alone
	#print("Mean squared error for {%.2f} training data and {%.2f} test data is {%.4f}.".format(training_percentage, testing_percentage, mse))

	return merge_compare