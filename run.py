import sys
import json
import os
import pandas as pd

sys.path.insert(0, 'src')
 
import datasets.make_dataset
from datasets.make_dataset import get_data


def clean_prev(cwd):
    files_to_remove = []

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

    # NEED TO CHANGE FOR NEW WORK
    #if os.path.exists(cwd + "/test/testdata/" + 'test_results_cooling_thermal_power.csv'):
    #    files_to_remove.append(cwd + "/test/testdata/" + 'test_results_cooling_thermal_power.csv')

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
    dataset = None
    out_data_stem = None
    out_file = None

    if 'clean' in targets:
        print('clean was specified: previous results are being removed')
        clean_prev(cwd)
        print('finished cleaning')

    if 'data' in targets:
        print('in run -> data')

        with open('config/data_params.json') as fh:
            data_cfg = json.load(fh)

        if not os.path.isdir(cwd + data_cfg['data_folder']):
            os.mkdir(cwd + data_cfg['data_folder'])

        dataset = get_data(cwd, **data_cfg)

        # make the data target
        out_data_stem = "/data/out/"

        if not os.path.isdir(cwd + out_data_stem):
            os.mkdir(cwd + out_data_stem)

        out_file = out_data_stem + 'cooling_thermal_power.csv'
        
    # from Quarter 1 - need to adapt
    #elif 'test' in targets:
    #    out_data_stem = "/test/testdata/"
    #    dataset = pd.read_csv(cwd + out_data_stem + 'test_data.csv')
    #    out_file = out_data_stem + 'test_results_cooling_thermal_power.csv'

    if 'features' in targets:
        print('in run -> features')
        print("features not finished yet")
        #print("This model doesn't distinguish separate features from the model because there are no significant transformations.")

    # from Quarter 1 - need to adapt
    if 'model' in targets:
        print("model not finished yet")
    #    print("in run -> model")
    #    with open('config/model_params.json') as fh:
        #     model_cfg = json.load(fh)

        # final_data = train_model(dataset, model_cfg)
        # final_data.to_csv(cwd + out_file)
        # print('Values saved to: ' + cwd + out_file)

        # # Compares by rerunning - rounds to prevent roundoff error
        # check_data = pd.read_csv(cwd + out_file, index_col = 0)['cooling_thermal_power'].round(2)

        # if final_data.round(2).equals(check_data):
        #     print('Data reread from file to confirm equality: data has been generated and saved correctly.')
        # else:
        #     print('Data reread from file to confirm equality: was not equal - please check to see if an error was raised.')



if __name__ == '__main__':
    # run via:
    # python run.py data model
    # or python run.py test model

    # clean via:
    # python run.py clean

    targets = sys.argv[1:]

    if 'all' in targets:
        targets = ['data', 'features', 'model']
    print(targets)

    if 'test' in targets:
        targets.append('model')

    if 'data' not in targets and 'test' not in targets:
        print('No valid data was specified (with either "data" or "test"), so an error will be raised if you included the model keyword.')

    main(targets)
    print('finished running')