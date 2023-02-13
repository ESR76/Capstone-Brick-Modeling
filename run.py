import sys
import json
import os
import pandas as pd

sys.path.insert(0, 'src')
 
import datasets.make_dataset
from datasets.make_dataset import get_data

import features.build_features
from features.build_features import time_features

import models.tree_model
from models.tree_model import generate_model

# function called for cleaning data
def clean_prev(cwd):
    print('in run -> clean')
    print('clean was specified: previous model and test results are being removed')
    files_to_remove = []
    pathways = ['/data/raw/', '/data/temp/', '/data/out/']

    # removing all data autodownloaded/generated
    if os.path.isdir(cwd + '/data/'):
        for pathway in pathways:
            if os.path.isdir(cwd + pathway):
                files = os.listdir(cwd + pathway)
                for file in files:
                    files_to_remove.append(cwd + pathway + file)

    # Test files
    test_files = os.listdir(cwd + '/test/' + 'testdata/')
    test_files.remove('test_data.csv')

    for i, file in enumerate(test_files):
        new_file = cwd + '/test/' + 'testdata/' + file
        test_files[i] = new_file

    files_to_remove.extend(test_files)

    for file in files_to_remove:
        os.remove(file)

    print('finished cleaning')
    return

# function for running test case
def test(cwd):
    print('in run -> test')
    print('Will run the current process on a test subset of data: features -> model.')

    with open('config/test_params.json') as fh:
        test_cfg = json.load(fh)

    # data
    early_dataset = pd.read_csv(cwd + test_cfg['test_directory'] + test_cfg['orig_name'], index_col = 0)
    # features
    finished_dataset = time_features(cwd, early_dataset, False, **test_cfg)
    # model
    modeled_predictions = generate_model(cwd, finished_dataset, False, **test_cfg)

    print('finished with test')
    return

# function for running current modeling steps
def data(cwd):
    print('in run -> data')
    with open('config/data_params.json') as fh:
        data_cfg = json.load(fh)

    if not os.path.isdir(cwd + data_cfg['data_folder']):
        os.mkdir(cwd + data_cfg['data_folder'])

    return get_data(cwd, **data_cfg)

def features(cwd, ds):
    print('in run -> features')

    with open('config/features_params.json') as fh:
        features_cfg = json.load(fh)

    if ds.empty:
        print('data was not in call to run.py file - will pull data from data/temp assuming data has been run before. Will raise error if data files never generated.')
        ds = pd.read_csv(cwd + features_cfg['temp_output'] + features_cfg['inter_name'], index_col = 0)

    return time_features(cwd, ds, True, **features_cfg)

def model(cwd, ds):
    print("in run -> model")

    with open('config/model_params.json') as fh:
        model_cfg = json.load(fh)

    if ds.empty:
        print('features was not in call to run.py file - will pull data from data/temp assuming features has been run before. Will raise error if features file never generated.')
        ds = pd.read_csv(cwd + model_cfg['temp_output'] + model_cfg['pre_model_name'], index_col = 0)

        ds[model_cfg['timestamp_col_tree']] = ds[model_cfg['timestamp_col_tree']].transform(pd.Timestamp)

    return generate_model(cwd, ds, True, **model_cfg)

def visualize(cwd, ds):
    print('in run -> visualize')
    print('visualize has not been defined yet.')
    return

def main(targets):
    '''
        Runs the main project pipeline logic, given the targets.
        targets must contain: 'data', 'model'.
        `main` runs the targets in order of data=>model.
    '''
    order = []

    cwd = os.getcwd()

    # runs clean before running anything else
    if 'clean' in targets:
        clean_prev(cwd)
        order.append('clean')

    # runs test before running any pipeline state
    if 'test' in targets:
        test(cwd)
        order.append('test')

    # runs data, features, and model in order - if trying to run without the other, it will print a statement and assume others have been run before
    early_dataset = pd.DataFrame()
    if 'data' in targets:
        early_dataset = data(cwd)
        order.append('data')
        
    finished_dataset = pd.DataFrame()
    if 'features' in targets:
        finished_dataset = features(cwd, early_dataset)
        order.append('features')
        
    modeled_dataset = pd.DataFrame()
    if 'model' in targets:
        modeled_dataset = model(cwd, finished_dataset)
        order.append('model')

    if 'visualize' in targets:
        visualize(cwd, modeled_dataset)
        order.append('visualize')
    
    return order


if __name__ == '__main__':
    # run via:
    # python run.py data features model or run.py all
    
    # test via:
    # python run.py test

    # clean via:
    # python run.py clean

    targets = sys.argv[1:]

    if 'all' in targets:
        targets.extend(['data', 'features', 'model'])
        targets.remove('all')

    run_order = main(targets)
    print('Function call finished running. Order of calls performed was: ' + ", ".join(run_order) + ".")


