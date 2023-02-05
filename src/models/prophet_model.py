import prophet as Prophet
import pandas as pd
import numpy as np


def generate_model(cwd, data, is_train, **params):
	print("in model..")

	dates = data['time_transformed']

	dates_train = (dates < pd.Timestamp('2018-08-01'))

	training_data = data.loc[dates_test, :].reset_index(drop = True).drop(['time_transformed'], axis = 1)
	training_percentage = training_data.shape[0] / data.shape[0] * 100
	testing_data = data.loc[~dates_test, :].reset_index(drop = True)
	testing_percentage = testing_data.shape[0] / data.shape[0] * 100

	model = Prophet.Prophet()
	model.fit(training_data)
	prophet_forecast = model.make_future_dataframe(periods = 51000, freq = '5min')
	prophet_forecast = model.predict(prophet_forecast)

	prophet_forecast_reduced = prophet_forecast.loc[:, ['ds', 'yhat']]
	prophet_forecast_reduced['time_transformed'] = prophet_forecast_reduced['ds'].transform(lambda x: pd.Timestamp(x))

	merge_compare = prophet_forecast_reduced.merge(testing_data, left_on = 'time_transformed', right_on = 'time_transformed')

	mse = ((merge_compare['yhat'] - merge_compare['y']) ** 2).sum() / merge_compare.shape[0]

	print("Mean squared error for %{0.2f} training data and %{0.2f} test data is {0.4f}.".format(training_percentage, testing_percentage, mse))

	return