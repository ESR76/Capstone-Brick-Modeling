import pandas as pd
import os
import requests
import csv

def combine_floor_csvs(dfs, out_fp):
	combined_df = pd.concat(dfs).reset_index(drop = True)

	if not os.path.isdir(out_fp):
		os.mkdir(out_fp)

	combined_df.to_csv(out_fp + 'combined_floors.csv')

	return combined_df

def file_download(raw_path, url, filename):
	# Adapted from Stack Overflow results
	response = requests.get(url)

	if response.status_code == 200:
		with open(raw_path + filename, 'w') as f:
			writer = csv.writer(f)
			for line in response.iter_lines():
				writer.writerow(line.decode('utf-8').split(','))
	else:
		print("Could not download " + filename + ". Ran into status_code: " + str(response.status_code))

	return

def get_floor_csvs(raw_path, temp_path, check_files, col_list, file_names):
	dfs = []

	for file_name in file_names:
		if file_name not in check_files:
			print(file_name + ' not present in raw data files: starting data download.')
			url = 'https://raw.githubusercontent.com/HYDesmondLiu/B2RL/master/real_building_buffers/{}'.format(file_name)
			file_download(raw_path, url, file_name)
			print('finished with ' + file_name + ' data')
		df = pd.read_csv(raw_path + file_name, usecols = col_list)
		if '2F' in file_name:
			df['floor'] = 2
		elif '3F' in file_name:
			df['floor'] = 3
		elif '4F' in file_name:
			df['floor'] = 4
		dfs.append(df)

	return combine_floor_csvs(dfs, temp_path)

def get_data(cwd, **params):
	print("in data..")
	files = []
	if os.path.isdir(cwd + params['raw_output']):
		files = os.listdir(cwd + params['raw_output'])
	else:
		os.mkdir(cwd + params['raw_output'])
	dataset = get_floor_csvs(cwd + params['raw_output'], cwd + params['temp_output'], files, params['col_list'], params['file_names'])
	return dataset