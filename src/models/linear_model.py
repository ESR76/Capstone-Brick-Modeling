#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# NOW CLEANED FOR PIPELINE LOGIC

def temp_conversion(val):
    return (val - 32) * 5/9 + 273.15

def generate_model_lin(cwd, data, is_train, **params):
    print("in model..")

    train_name = params['train_data']
    test_name = params['test_data']
    final_name = params['modeled_preds']
    output_col = params['output_col']

    if is_train:
        if os.path.isdir(cwd + params['final_output']):
            files = os.listdir(cwd + params['final_output'])

            if final_name in files:
                print('Modeled data already found - regenerating because of model call.')
        else:
            os.mkdir(cwd + params['final_output'])
    else:
        print("in run -> model")

    training_data = data.loc[data.loc[:, "train"], :].reset_index(drop = True).drop(["train", "imputed"], axis = 1)
    testing_data = data.loc[~data.loc[:, "train"], :].reset_index(drop = True).drop(["train", "imputed"], axis = 1)

    regr = LinearRegression()
    
    # month gave worse results in testing so it's been removed
    Xtrain = training_data.drop([output_col, 'month'], axis = 1)
    Ytrain = training_data[output_col]

    Xtest = testing_data.drop([output_col, 'month'], axis = 1)
    Ytest = testing_data[output_col]

    regr.fit(Xtrain, Ytrain)
    y_pred = regr.predict(Xtest)
    pred_series = pd.Series(y_pred).rename("preds")
    mse = mean_squared_error(np.array(Ytest), y_pred)

    print("*****")
    print('MSE: {0:.4f}.'.format(mse))
    print("*****")
    
    # if is_train:
    #     training_data.to_csv(cwd + params['final_output'] + train_name, index = False)
    #     testing_data.to_csv(cwd + params['final_output'] + test_name, index = False)

    #     pred_series.to_csv(cwd + params['final_output'] + final_name, index = False)
    # else:
    #     training_data.to_csv(cwd + params['test_directory'] + train_name)
    #     testing_data.to_csv(cwd + params['test_directory'] + test_name)

    #     pred_series.to_csv(cwd + params['test_directory'] + final_name)

    return pred_series