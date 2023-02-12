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


def clean_prev(cwd):
    files_to_remove = []

    # removing all directories
    if os.path.isdir(cwd + '/data/'):
        if os.path.isdir(cwd + '/data/' + 'raw/'):
            files = os.listdir(cwd + '/data/' + 'raw/')
            for file in files:
                files_to_remove.append(cwd + '/data/' + 'raw/' + file)
        if os.path.isdir(cwd + '/data/' + 'temp/'):
            files = os.listdir(cwd + '/data/' + 'temp/')
            for file in files:
                files_to_remove.append(cwd + '/data/' + 'temp/' + file)
        if os.path.isdir(cwd + '/data/' + 'out/'):
            files = os.listdir(cwd + '/data/' + 'out/')
            for file in files:
                files_to_remove.append(cwd + '/data/' + 'out/' + file)

    # Test files
    test_files = os.listdir(cwd + '/test/' + 'testdata/')
    test_files.remove('test_data.csv')

    for i, file in enumerate(test_files):
        new_file = cwd + '/test/' + 'testdata/' + file
        test_files[i] = new_file

    files_to_remove.extend(test_files)

    for file in files_to_remove:
        os.remove(file)

    return


def main(targets):
    '''
        Runs the main project pipeline logic, given the targets.
        targets must contain: 'data', 'model'.
        `main` runs the targets in order of data=>model.
    '''

    cwd = os.getcwd()
    early_dataset = pd.DataFrame()

    if 'clean' in targets:
        print('clean was specified: previous model and test results are being removed')
        clean_prev(cwd)
        print('finished cleaning')

    if 'test' in targets:
        print('in run -> test')
        print('Will run the whole process on a test subset of data, from feature creation to model.')

        with open('config/test_params.json') as fh:
            test_cfg = json.load(fh)

        # data
        early_dataset = pd.read_csv(cwd + test_cfg['test_directory'] + test_cfg['orig_name'], index_col = 0)
        # features
        finished_dataset = time_features(cwd, early_dataset, False, **test_cfg)
        # model
        modeled_predictions = generate_model(cwd, finished_dataset, False, **test_cfg)

    if 'data' in targets:
        print('in run -> data')

        with open('config/data_params.json') as fh:
            data_cfg = json.load(fh)

        if not os.path.isdir(cwd + data_cfg['data_folder']):
            os.mkdir(cwd + data_cfg['data_folder'])

        early_dataset = get_data(cwd, **data_cfg)

        # make the data target
        #out_data_stem = data_cfg['final_output']

        # Makes out data if needed
        
        #if not os.path.isdir(cwd + out_data_stem):
        #    os.mkdir(cwd + out_data_stem)

    finished_dataset = pd.DataFrame()

    if 'features' in targets:
        print('in run -> features')

        with open('config/features_params.json') as fh:
            features_cfg = json.load(fh)

        if early_dataset.empty:
            print('data was not in call to run.py file - will pull data from data/temp assuming data has been run before. Will raise error if data files never generated.')
            early_dataset = pd.read_csv(cwd + features_cfg['temp_output'] + features_cfg['inter_name'], index_col = 0)

        finished_dataset = time_features(cwd, early_dataset, True, **features_cfg)

    modeled_dataset = pd.DataFrame()

    if 'model' in targets:
        print("in run -> model")

        with open('config/model_params.json') as fh:
            model_cfg = json.load(fh)

        if finished_dataset.empty:
            print('features was not in call to run.py file - will pull data from data/temp assuming features has been run before. Will raise error if features file never generated.')
            finished_dataset = pd.read_csv(cwd + model_cfg['temp_output'] + model_cfg['pre_model_name'], index_col = 0)

            finished_dataset[model_cfg['timestamp_col_prophet']] = finished_dataset[model_cfg['timestamp_col_prophet']].transform(pd.Timestamp)


        modeled_predictions = generate_model(cwd, finished_dataset, True, **model_cfg)



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

    main(targets)
    print('finished running')