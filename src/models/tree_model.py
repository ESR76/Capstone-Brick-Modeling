from sklearn import tree
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


def generate_model(cwd, data, is_train, **params):
	print("in model..")
	timestamp_col = params["timestamp_col_prophet"]

	dates = data[timestamp_col]

	dates_train = (dates < pd.Timestamp(params["split_date"]))

	training_data = data.loc[dates_train, :].reset_index(drop = True).drop([timestamp_col], axis = 1)
	training_percentage = training_data.shape[0] / data.shape[0] * 100
	testing_data = data.loc[~dates_train, :].reset_index(drop = True)
	testing_percentage = testing_data.shape[0] / data.shape[0] * 100

	clf = tree.DecisionTreeRegressor(max_depth = 7, min_samples_split = 5)
	
	Ytrain = dates.loc[dates_train, :].reset_index(drop = True)
	
	Ytest = dates.loc[~dates_train, :].reset_index(drop = True)

	clf = clf.fit(training_data, Ytrain)
	y_pred = clf.predict(testing_data)
	mse = mean_squared_error(Ytrain,y_pred)

	print('Training percentage: ' + str(training_percentage) + ', testing percentage: ' + str(testing_percentage) + ', prediction mean squared error: ' + str(mse))
	
	# not sure what I'm doing wrong with this line right now but leaving it alone
	#print("Mean squared error for {%.2f} training data and {%.2f} test data is {%.4f}.".format(training_percentage, testing_percentage, mse))

	return y_pred