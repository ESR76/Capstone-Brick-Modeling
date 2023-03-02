import pandas as pd
import os
import requests
import csv

def combine_floor_csvs(dfs, out_fp, out_name):
	combined_df = pd.concat(dfs).reset_index(drop = True)

	if not os.path.isdir(out_fp):
		os.mkdir(out_fp)

	combined_df.to_csv(out_fp + out_name, index = False)

	return combined_df

def file_download(out_path, url, filename):
	# Adapted from Stack Overflow results
	response = requests.get(url)

	if response.status_code == 200:
		with open(out_path + filename, 'w') as f:
			writer = csv.writer(f)
			for line in response.iter_lines():
				writer.writerow(line.decode('utf-8').split(','))
	else:
		print("Could not download " + filename + ". Ran into status_code: " + str(response.status_code))

	return

def get_floor_csvs(in_path, out_path, check_files, col_list, file_names, out_name, add_floor_names):
	dfs = []

	for file_name in file_names:
		if file_name not in check_files:
			print(file_name + ' not present in raw data files: starting data download.')
			url = 'https://raw.githubusercontent.com/HYDesmondLiu/B2RL/master/real_building_buffers/{}'.format(file_name)
			file_download(in_path, url, file_name)
			print('finished with ' + file_name + ' data')
		else:
			print(file_name + ' present in raw data files: skipping download.')
		# limiting columns for consistency
		df = pd.read_csv(in_path + file_name, usecols = col_list)
		if '2F' in file_name and add_floor_names:
			df.loc[:, 'floor'] = 2
		elif '3F' in file_name and add_floor_names:
			df.loc[:, 'floor'] = 3
		elif '4F' in file_name and add_floor_names:
			df.loc[:, 'floor'] = 4
		dfs.append(df)

	return combine_floor_csvs(dfs, out_path, out_name)

def get_data(cwd, **params):
	files = []
	if not os.path.isdir(cwd + params['raw_output']):
		os.mkdir(cwd + params['raw_output'])
	else:
		files = os.listdir(cwd + params['raw_output'])

	if os.path.isdir(cwd + params['temp_output']):
		files = os.listdir(cwd + params['temp_output'])
		if params['out_name'] in files:
			print('All data already downloaded and ready for features - will not redevelop for time saving.')
			print('To redownload, please run "python3 run.py clean" and then call data again.')
			return pd.read_csv(cwd + params['temp_output'] + params['out_name'])

	# False for add_floor_names because cleaning version we're using does not use it or gain information from it
	dataset = get_floor_csvs(cwd + params['raw_output'], cwd + params['temp_output'], files, params['col_list'], params['file_names'], params['out_name'], False)
	return dataset