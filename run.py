# IMPORTS FOR MAIN CODE
import sys
import json
import os
import pandas as pd
import warnings
from sklearn import tree

# IMPORTS FOR SCRIPT
sys.path.insert(0, 'src')
 
from datasets.make_dataset import *

from features.build_features import *
from features.clean_features import *

from models.tree_model import *

from optimization.optimize import *

from visualization.visualize import *

# function called for cleaning data
def clean_prev(cwd):
    print('\nin run -> clean repo')
    print('clean was specified: previous model and test results are being removed')
    files_to_remove = []
    pathways = ['/data/raw/', '/data/temp/', '/data/out/']

    # removing all regular data autodownloaded/generated
    if os.path.isdir(cwd + '/data/'):
        for pathway in pathways:
            if os.path.isdir(cwd + pathway):
                files = os.listdir(cwd + pathway)
                for file in files:
                    if file != 'output_optsets':
                        files_to_remove.append(cwd + pathway + file)

    # Test files
    # don't need to check if test data is directory because it exists upon cloning
    test_files = os.listdir(cwd + '/test/' + 'testdata/')
    test_files.remove('test_data.csv')

    if 'output_optsets' in test_files:
        test_files.remove('output_optsets')

    for i, file in enumerate(test_files):
        new_file = cwd + '/test/' + 'testdata/' + file
        test_files[i] = new_file

    files_to_remove.extend(test_files)

    # optimize files
    if os.path.isdir(cwd + '/data/out/output_optsets/'):
        optfiles = os.listdir(cwd + '/data/out/output_optsets/')
        for file in optfiles:
            files_to_remove.append(cwd + '/data/out/output_optsets/' + file)

    # Test optimize files
    if os.path.isdir(cwd + '/test/testdata/output_optsets/'):
        optfiles = os.listdir(cwd + '/test/testdata/output_optsets/')
        for file in optfiles:
            files_to_remove.append(cwd + '/test/testdata/output_optsets/' + file)

    if os.path.isdir(cwd + '/visualizations/'):
        files = os.listdir(cwd + '/visualizations/')
        for file in files:
            files_to_remove.append(cwd + '/visualizations/' + file)

    if os.path.isdir(cwd + '/test/testviz/'):
        files = os.listdir(cwd + '/test/testviz/')
        for file in files:
            files_to_remove.append(cwd + '/test/testviz/' + file)

    for file in files_to_remove:
        os.remove(file)

    print('finished cleaning')
    return

# function for running test case
def test(cwd):
    print('\nin run -> test')
    print('Will run the current process on a test subset of data: features (pt. 1 and 2) -> model -> optimize -> visualize.')

    with open('config/test_params.json') as fh:
        test_cfg = json.load(fh)

    # data
    early_dataset = pd.read_csv(cwd + test_cfg['test_directory'] + test_cfg['orig_name'], index_col = 0).drop(['floor'], axis = 1)
    # features
    cleaned_dataset = clean_raw(cwd, early_dataset, False, **test_cfg)
    finished_dataset = time_features(cwd, cleaned_dataset, False, **test_cfg)
    # model
    test_mdl = generate_model(cwd, finished_dataset, False, **test_cfg)
    # optimize
    output_optimize = optimize_model(cwd, test_mdl, False, **test_cfg)
    # visualize
    visualize_text = visualize_results(cwd, output_optimize, False, **test_cfg)
    # STILL FILLING IN

    print('finished with test')
    return

# function for running current modeling steps
def data(cwd):
    print('\nin run -> data')
    with open('config/data_params.json') as fh:
        data_cfg = json.load(fh)

    if not os.path.isdir(cwd + data_cfg['data_folder']):
        os.mkdir(cwd + data_cfg['data_folder'])

    return get_data(cwd, **data_cfg)

def features_1(cwd, ds):
    print('\nin run -> features pt. 1: cleaning raw data')

    with open('config/clean_params.json') as fh:
        clean_cfg = json.load(fh)

    if ds.empty:
        print('data was not in call to run.py file - will pull data from data/temp assuming data has been run before. Will raise error if data files never generated.')
        ds = pd.read_csv(cwd + clean_cfg['temp_output'] + clean_cfg['in_name'])

    return clean_raw(cwd, ds, True, **clean_cfg)

def features_2(cwd, ds):
    print('\nin run -> features pt.2: generating features for model')

    with open('config/features_params.json') as fh:
        features_cfg = json.load(fh)

    # no warning because features_2 runs after features_1

    return time_features(cwd, ds, True, **features_cfg)

def model(cwd, ds):
    print("\nin run -> model")

    with open('config/model_params.json') as fh:
        model_cfg = json.load(fh)

    if ds.empty:
        print('features was not in call to run.py file - will pull data from data/temp assuming features has been run before. Will raise error if features file never generated.')
        ds = pd.read_csv(cwd + model_cfg['temp_output'] + model_cfg['pre_model_name'])

    return generate_model(cwd, ds, True, **model_cfg)

def optimize(cwd, mdl):
    # pulls optimize params
    with open('config/optimize_params.json') as fh:
        optimize_cfg = json.load(fh)

    if mdl is None:
        print('Optimize assumes that you have run the model part of the pipeline at least so it will raise an error if you have not yet done so.')
        print('The model keyword will be rerun briefly to grab the trained model to be used in the optimization.')
        mdl = model(cwd, pd.DataFrame())

    print('\nin run -> optimize')

    return optimize_model(cwd, mdl, True, **optimize_cfg)

def visualize(cwd, ds):
    print('\nin run -> visualize')

    # pulls visualize params
    with open('config/visualize_params.json') as fh:
        visualize_cfg = json.load(fh)

    if ds.empty:
        print('optimize was not in call to run.py file - will pull data from data/out assuming optimize & rest of the data pipeline has been run before. Will raise error if features file never generated.')
        ds = pd.read_csv(cwd + visualize_cfg['final_output'] + visualize_cfg['optimize_results'])

    return visualize_results(cwd, ds, True, **visualize_cfg)

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
        cleaned_dataset = features_1(cwd, early_dataset)
        finished_dataset = features_2(cwd, cleaned_dataset)
        order.append('features')
        
    mdl = None
    if 'model' in targets:
        mdl = model(cwd, finished_dataset)
        order.append('model')

    opt_results = pd.DataFrame()
    if 'optimize' in targets:
        opt_results = optimize(cwd, mdl)
        order.append('optimize')

    if 'visualize' in targets:
        print(visualize(cwd, opt_results))
        order.append('visualize')

    return order


if __name__ == '__main__':
    # run via:
    # python run.py data features model optimize or run.py all
    
    # test via:
    # python run.py test

    # clean via:
    # python run.py clean

    targets = sys.argv[1:]
    warnings.filterwarnings('ignore')

    if 'all' in targets:
        targets.extend(['data', 'features', 'model', 'optimize', 'visualize'])
        targets.remove('all')

    run_order = main(targets)
    print('Function call finished running. Order of calls performed was: ' + ", ".join(run_order) + ".")


