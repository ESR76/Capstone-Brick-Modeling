import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# should've changed the data but this is next best option - changing for visual only
def visualize_hour(x):
	replace = x - 8
	if replace < 0:
		replace = 24 + replace
	return replace

def prop_zeros(x):
	prop = 0

	if len(x) > 0:
		prop = (sum(x == 0) / len(x))

	return prop

def visualize_results(cwd, opt_results, is_train, **params):
	save_loc = cwd + params['save_viz']
	direc = ""
	files_created = []
	final_name = params['is_visualized']

	# change to false when testing visuals - TO DO
	skip_duplicates = False

	if not os.path.isdir(save_loc):
		os.mkdir(save_loc)

	if is_train:
		direc = params['temp_output']

		if skip_duplicates:
			files = os.listdir(cwd + params['final_output'])

			if final_name in files:
				print('Visualize files already found according to the visualize_complete text file.')
				print('To regenerate - please run "python3 run.py clean" before calling visualize again.')
					
				return 'Visualizations not recreated, please check your visualizations folder.'
	else:
		print("\nin run -> visualize for test data")
		direc = params['test_directory']


	### VISUALIZATION OF TRENDS BY UNIT OF TIME
	print('Visualizing to folder: {}'.format(save_loc))
	plot_df = pd.read_csv(cwd + direc + params['orig_name'])

	plot_df.loc[:, params['time_col']] = plot_df.loc[:, params['time_col']].transform(pd.Timestamp)

	plot_df.loc[:, 'hour'] = plot_df.loc[:, params['time_col']].transform(lambda x: visualize_hour(x.hour))

	plot_df.loc[:, 'weekday'] = plot_df.loc[:, params['time_col']].transform(lambda x: x.weekday)
	plot_df.loc[:, 'weekday'] = plot_df.loc[:, 'weekday'].replace({0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thurs', 4: 'Fri', 5: 'Sat', 6: 'Sun'})
	plot_df.loc[:, 'weekday'] = pd.Categorical(plot_df.loc[:, 'weekday'], categories = ["Sun", "Mon", "Tues", "Wed", "Thurs", "Fri", "Sat"], ordered = True)

	# Currently unused
	# plot_df.loc[:, 'month'] = plot_df.loc[:, params['time_col']].transform(lambda x: x.month)
	# plot_df.loc[:, 'month'] = plot_df.loc[:, 'month'].replace({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
	# plot_df.loc[:, 'month'] = pd.Categorical(plot_df.loc[:, 'month'], categories = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], ordered = True)

	# month_groups = plot_df.groupby('month')['energy'].mean()
	weekday_groups = plot_df.groupby('weekday')['energy'].mean()
	hour_groups = plot_df.groupby('hour')['energy'].mean()

	fig, axes = plt.subplots(nrows=1, ncols=2, dpi=300, figsize=(12,6))

	sns.lineplot(data = hour_groups.reset_index(), x = 'hour', y = 'energy', ax = axes[0])
	axes[0].set_title('Hour Means for Energy')
	sns.lineplot(data = weekday_groups.reset_index(), x = 'weekday', y = 'energy', ax = axes[1], sort = False)
	axes[1].set_title('Weekday Means for Energy')

	fig.suptitle('Energy Means by Different Time Groups')
	plt.tight_layout()
	plt.savefig(save_loc + 'energy_time_means.png')
	plt.clf()
	files_created.append('energy_time_means.png')

	# Visualization of 0 proportions by hour
	hour_0s = plot_df.groupby('hour')['energy'].agg(prop_zeros).reset_index()

	if is_train:
		bar_0s = sns.barplot(data=hour_0s, x="hour", y="energy")
		bar_0s.set(xlabel = 'Hour', ylabel = 'Proportion of Zero Values', title = 'Proportion of Zero Values by Hour')
		fig = bar_0s.get_figure()
		plt.savefig(save_loc + 'dataset_bar_0s.png', bbox_inches='tight', s = 20, dpi = 300, figsize = (12,12))
		plt.clf()
		files_created.append('dataset_bar_0s.png')
	else:
		fig, axes = plt.subplots(figsize=(12,12))
		bar_0s = sns.barplot(data=hour_0s, x="hour", y="energy", ax = axes)
		bar_0s.set(xlabel = 'Hour', ylabel = 'Proportion of Zero Values', title = 'Proportion of Zero Values by Hour')
		axes2 = axes.twinx()
		line_0s = sns.lineplot(data=hour_0s, x=axes.get_xticks(), y="energy", ax = axes2)
		line_0s.set(xlabel = 'Hour', ylabel = 'Proportion of Zero Values', title = 'Proportion of Zero Values by Hour')

		plt.savefig(save_loc + 'dataset_bar_0s.png', bbox_inches='tight', s = 20, dpi = 300, figsize = (12,12))
		plt.clf()
		files_created.append('dataset_bar_0s.png')
		print('Test data for dataset_bar_0s.png has no 0 data, so a line was super imposed on top to show that the proportions were 0.')

	plot_df = pd.read_csv(cwd + direc + params['out_name'])

	plot_df.loc[:, params['time_col']] = plot_df.loc[:, params['time_col']].transform(pd.Timestamp)
	# only looking at non-imputed data
	plot_df = plot_df.loc[~plot_df['imputed'], :]

	#print(plot_df.head(5))
	vis_columns = params['viz_columns']
	vis_rename = params['viz_rename']
	vis_dict = dict(zip(vis_columns, vis_rename))

	fig, axes = plt.subplots(nrows=2, ncols=3, dpi=300, figsize=(18,6))
	for i, ax in enumerate(axes.flatten()):
		data = plot_df.loc[:, [vis_columns[0], vis_columns[i + 1]]]
		ax.plot(data.loc[:, vis_columns[0]], data.loc[:, vis_columns[i + 1]], color='red')
		# Decorations
		title = vis_dict[vis_columns[i + 1]]
		if len(title) == 0:
			ax.set_title(vis_columns[i + 1])
		else:
			ax.set_title(title)
		ax.xaxis.set_ticks_position('none')
		ax.yaxis.set_ticks_position('none')
		ax.spines["top"].set_alpha(0)
		ax.tick_params(labelsize=6)
		ax.tick_params(axis='x', labelrotation = 45)

	plt.tight_layout()
	plt.savefig(save_loc + 'data_values_pre_imputation.png')
	plt.clf()
	files_created.append('data_values_pre_imputation.png')

	rel_fig = sns.regplot(data = plot_df, x = 'Actual Sup Flow SP' , y = 'Actual Supply Flow')
	rel_fig.set(xlabel = 'Airflow SP', ylabel = 'Actual Airflow', title = 'Residuals of Regression on Actual Airflow vs Setpoint')
	fig = rel_fig.get_figure()
	plt.savefig(save_loc + 'data_airflow_reg.png', bbox_inches='tight', s = 20, dpi = 300, figsize = (12,12))
	plt.clf()
	files_created.append('data_airflow_reg.png')

	### VISUALIZATION of OPTIMIZATION RESULTS ###

	opt_results.loc[:, 'hour'] = opt_results.loc[:, 'hour'].transform(visualize_hour)


	# CURRENTLY UNUSED
	# unique_air_vals = params['optimize_options']['Actual Sup Flow SP']
	# num_groups = len(unique_air_vals)

	# for measurement in ['median', 'mean', 'min', 'max']:
	# 	fig, axes = plt.subplots(nrows= (num_groups + 1) // 2, ncols=2, dpi=300, figsize=(18,18))
	# 	row = 0
	# 	col = 0
	# 	for i in range(num_groups):
	# 		val = unique_air_vals[i]
	# 		subset = opt_results.loc[opt_results['air_set'] == val, :]
	# 		sns.violinplot(data = subset, x = measurement, y = 'occupancy', ax = axes[row, col])
	# 		axes[row, col].set(title = 'Violin Plot of {0} Differences at Air Setpoint = {1}'.format(measurement.title(), val))

	# 		col = col + 1
	# 		if col == 2:
	# 			row += 1
	# 			col = 0

	# 	fig.suptitle('Violin Plots of Differences')
	# 	plt.tight_layout()
	# 	plt.savefig((save_loc + 'opt_results_violin_{}.png').format(measurement), bbox_inches='tight')
	# 	plt.clf()
	# 	files_created.append('opt_results_violin_{}.png'.format(measurement))

	heat_df = pd.pivot_table(opt_results, values='max', index=['hour'],
                    columns=['air_set'], aggfunc=max)

	heat_fig = sns.heatmap(data = heat_df, annot = True)
	heat_fig.set(xlabel = 'Air Setpoint Reduction', ylabel = 'Hour', title = 'Maximum Reduction of Cost by Air Setpoint Reduction and Hour')
	fig = heat_fig.get_figure()
	plt.savefig(save_loc + 'opt_results_heat_max.png', bbox_inches='tight', s = 20, dpi = 300, figsize = (12,12))
	plt.clf()
	files_created.append('opt_results_heat_max.png')

	plot_df = opt_results.loc[(opt_results['temp_set'] == 0), ['hour', 'median', 'air_set', 'occupancy']]
	order = ['high', 'low', 'unoccupied']
	bar_fig = sns.barplot(data = plot_df, x = 'hour', y = 'median', hue = 'occupancy', hue_order = order)
	bar_fig.set(xlabel = 'Hour', ylabel = 'Median Difference', title = 'Median Difference by Hour & Occupancy')
	fig = bar_fig.get_figure()
	fig.savefig(save_loc + 'opt_results_bar_occ.png', bbox_inches='tight', dpi = 300, figsize = (12, 12), linewdith = 30)
	plt.clf()

	bar_fig = sns.barplot(data = plot_df, x = 'air_set', y = 'median', hue = 'occupancy', hue_order = order)
	bar_fig.set(xlabel = 'Air Setpoint', ylabel = 'Median Difference', title = 'Median Difference by Air Setpoint & Occupancy')
	fig = bar_fig.get_figure()
	plt.legend(loc='upper left')
	fig.savefig(save_loc + 'opt_results_bar_occ_2.png', bbox_inches='tight', dpi = 300, figsize = (12, 12), linewdith = 30)
	plt.clf()

	## VISUALIZATION OF PROPORTIONS LIMITED ##

	if is_train:
		direc = params['final_output']
	total_results = pd.read_csv(cwd + direc + 'total_' + params['optimize_results'])

	total_results.loc[:, 'hour'] = total_results.loc[:, 'hour'].transform(visualize_hour)
	line_fig = sns.lineplot(data = total_results, x = 'hour', y = 'was_limited')
	line_fig.set(xlabel = 'Hour', ylabel = 'Proportion Limited', title = 'Proportion Limited by Occupancy Limits by Hour')
	fig = line_fig.get_figure()
	plt.savefig(save_loc + 'opt_results_limitations_hour.png', bbox_inches='tight', dpi = 300, figsize = (12, 12), linewdith = 30)
	plt.clf()
	files_created.append('opt_results_limitations_hour.png')

	if is_train:
		with open(cwd + params['final_output'] + params['is_visualized'], 'w') as f:
			f.write('Files created:\n')
			for file in files_created:
				f.write(file + '\n')
	else:
		with open(cwd + params['test_directory'] + params['is_visualized'], 'w') as f:
			f.write('Files created:\n')
			for file in files_created:
				f.write(file + '\n')

	return "Visualization complete. Thanks for running the pipeline!"