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

	if not os.path.isdir(save_loc):
		os.mkdir(save_loc)

	if is_train:
		direc = params['temp_output']

		files = os.listdir(cwd + params['final_output'])

		if final_name in files:
			print('Visualize files already found according to the visualize_complete text file.')
			print('To regenerate - please run "python3 run.py clean" before calling visualize again.')
				
			return 'Visualizations not printed, please check your visualizations folder.'
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

	plot_df.loc[:, 'month'] = plot_df.loc[:, params['time_col']].transform(lambda x: x.month)
	plot_df.loc[:, 'month'] = plot_df.loc[:, 'month'].replace({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
	plot_df.loc[:, 'month'] = pd.Categorical(plot_df.loc[:, 'month'], categories = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], ordered = True)

	#plot_df.loc[:, 'year'] = plot_df.loc[:, params['time_col']].transform(lambda x: x.year)
	#plot_df.loc[:, 'month/year'] = plot_df.apply(lambda x: str(x['month']) + '/' + str(x['year']), axis = 1)
	#mo_year_groups = plot_df.groupby('month/year')['energy'].mean()

	month_groups = plot_df.groupby('month')['energy'].mean()
	weekday_groups = plot_df.groupby('weekday')['energy'].mean()
	hour_groups = plot_df.groupby('hour')['energy'].mean()

	fig, axes = plt.subplots(nrows=1, ncols=2, dpi=300, figsize=(12,6))

	sns.lineplot(data = hour_groups.reset_index(), x = 'hour', y = 'energy', ax = axes[0])
	axes[0].set_title('Hour Means for Energy')
	sns.lineplot(data = weekday_groups.reset_index(), x = 'weekday', y = 'energy', ax = axes[1], sort = False)
	axes[1].set_title('Weekday Means for Energy')
	#sns.lineplot(data = month_groups.reset_index(), x = 'month', y = 'energy', ax = axes[2], sort = False)
	#axes[2].set_title('Month Means for Energy')

	fig.suptitle('Energy Means by Different Time Groups')
	plt.tight_layout()
	plt.savefig(save_loc + 'energy_time_means.png')
	plt.clf()
	files_created.append('energy_time_means.png')


	# Visualization of 0 proportions by hour
	hour_0s = plot_df.groupby('hour')['energy'].agg(prop_zeros).reset_index()

	if is_train:
		bar_0s = sns.barplot(data=hour_0s, x="hour", y="energy", ax = axes[0])
		bar_0s.set(xlabel = 'Hour', ylabel = 'Proportion of Zero Values', title = 'Proportion of Zero Values by Hour')
		fig = bar_0s.get_figure()
		plt.savefig(save_loc + 'opt_results_bar_0s.png', bbox_inches='tight', s = 20, dpi = 300, figsize = (12,12))
		plt.clf()
		files_created.append('opt_results_bar_0s.png')

	else:
		fig, axes = plt.subplots(figsize=(12,12))
		bar_0s = sns.barplot(data=hour_0s, x="hour", y="energy", ax = axes)
		bar_0s.set(xlabel = 'Hour', ylabel = 'Proportion of Zero Values', title = 'Proportion of Zero Values by Hour')
		axes2 = axes.twinx()
		line_0s = sns.lineplot(data=hour_0s, x=axes.get_xticks(), y="energy", ax = axes2)
		line_0s.set(xlabel = 'Hour', ylabel = 'Proportion of Zero Values', title = 'Proportion of Zero Values by Hour')

		plt.savefig(save_loc + 'opt_results_bar_0s.png', bbox_inches='tight', s = 20, dpi = 300, figsize = (12,12))
		plt.clf()
		files_created.append('opt_results_bar_0s.png')
		print('Test data for opt_results_bar_0s.png has no 0 data, so a line was super imposed on top to show that the proportions were 0.')

	### VISUALIZATION of DATA MEDIANS ###
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

	## OTHER VISUALIZATIONS ##

	opt_results.loc[:, 'hour'] = opt_results.loc[:, 'hour'].transform(visualize_hour)

	### VISUALIZATION of OPTIMIZATION RESULTS ###
	fig, axes = plt.subplots(nrows=2, ncols=2, dpi=300, figsize=(18,6))
	sns.histplot(opt_results.loc[:, 'mean'], ax = axes[0, 0])
	axes[0, 0].set(title = 'Mean Differences Histogram')

	sns.histplot(opt_results.loc[:, 'median'], ax = axes[0, 1])
	axes[0, 1].set(title = 'Median Differences Histogram')

	sns.histplot(opt_results.loc[:, 'min'], ax = axes[1, 0])
	axes[1, 0].set(title = 'Min Differences Histogram')

	sns.histplot(opt_results.loc[:, 'max'], ax = axes[1, 1])
	axes[1, 1].set(title = 'Max Differences Histogram')

	fig.suptitle('Histograms of Differences')
	plt.tight_layout()
	plt.savefig(save_loc + 'opt_results_differences.png', bbox_inches='tight')
	plt.clf()
	files_created.append('opt_results_differences.png')

	### VISUALIZATION of MEDIAN RESULTS ###
	fig, axes = plt.subplots(nrows=4, ncols=6, dpi=300, figsize=(24,24))
	row = 0
	col = 0
	for i in range(0, 24):
		subset = opt_results.loc[opt_results['hour'] == i, ['median']].reset_index(drop = True)

		if not subset.empty:
			sns.histplot(data = subset, x = 'median', ax = axes[row, col])
			axes[row, col].set(title = 'Histogram of Median Differences Hour {}'.format(i))

		col = col + 1
		if col == 6:
			row += 1
			col = 0

	fig.suptitle('Histograms of Differences')
	plt.tight_layout()
	plt.savefig(save_loc + 'opt_results_histplots_differences.png', bbox_inches='tight')
	plt.clf()
	files_created.append('opt_results_histplots_differences.png')

	### VISUALIZATION of MEDIAN RESULTS ###
	fig, axes = plt.subplots(nrows=4, ncols=6, dpi=300, figsize=(24,24))
	row = 0
	col = 0
	for i in range(0, 24):
		subset = opt_results.loc[opt_results['hour'] == i, :]
		sns.boxplot(data = subset, x = 'median', ax = axes[row, col])
		axes[row, col].set(title = 'Boxplot of Median Differences Hour {}'.format(i))

		col = col + 1
		if col == 6:
			row += 1
			col = 0

	fig.suptitle('Histograms of Differences')
	plt.tight_layout()
	plt.savefig(save_loc + 'opt_results_boxplots_differences_hour.png', bbox_inches='tight')
	plt.clf()
	files_created.append('opt_results_boxplots_differences_hour.png')

	# trying to get more info about low occupancy hours - examine relationships with prop boundary and hours TO DO

	scatter_fig = sns.scatterplot(data = opt_results, x = 'hour', y = 'median', hue = 'occupancy', s = 400, alpha = 0.5)
	fig = scatter_fig.get_figure()
	plt.legend(loc="upper right", frameon=True, fontsize=30)
	plt.savefig(save_loc + 'opt_results_scatter_median.png', bbox_inches='tight', s = 20, dpi = 300, figsize = (12,12))
	plt.clf()
	files_created.append('opt_results_scatter_median.png')

	swarm_fig = sns.swarmplot(data=opt_results, x="hour", y="median", hue="occupancy", s = 20)
	fig = swarm_fig.get_figure()
	plt.legend(loc="upper right", frameon=True, fontsize=30)
	plt.grid()
	plt.savefig(save_loc + 'opt_results_swarm_median.png', bbox_inches='tight', dpi = 300, figsize = (12,12))
	plt.clf()
	files_created.append('opt_results_swarm_median.png')

	fig, axes = plt.subplots(nrows=2, ncols=2, dpi=300, figsize=(18,18))
	sns.boxplot(data= opt_results, x="median", y="occupancy", ax = axes[0, 0])
	axes[0, 0].set(title = 'Median Differences Boxplot by Occupancy')

	sns.boxplot(data=opt_results, x="mean", y="occupancy", ax = axes[0, 1])
	axes[0, 1].set(title = 'Mean Differences Boxplot by Occupancy')

	sns.boxplot(data=opt_results, x="min", y="occupancy", ax = axes[1, 0])
	axes[1, 0].set(title = 'Minimum Differences Boxplot by Occupancy')

	sns.boxplot(data=opt_results, x="max", y="occupancy", ax = axes[1, 1])
	axes[1, 1].set(title = 'Max Differences Boxplot by Occupancy')

	fig.suptitle('Boxplots of Differences by Occupancy')
	plt.tight_layout()
	plt.savefig(save_loc + 'opt_results_box_groups.png', bbox_inches='tight')
	plt.clf()
	files_created.append('opt_results_box_groups.png')

	scatter_fig = sns.scatterplot(data = opt_results, x = 'hour', y = 'median', hue = 'occupancy', size = 'air_set', sizes = (20, 200), alpha = 0.5)
	fig = scatter_fig.get_figure()
	plt.legend(loc="upper right", frameon=True, fontsize=30)
	plt.savefig(save_loc + 'opt_results_scatter_air.png', bbox_inches='tight', s = 20, dpi = 300, figsize = (12,12))
	plt.clf()
	files_created.append('opt_results_scatter_air.png')

	heat_df = pd.pivot_table(opt_results, values='max', index=['hour'],
                    columns=['air_set'], aggfunc=max)

	heat_fig = sns.heatmap(data = heat_df, annot = True)
	heat_fig.set(xlabel = 'Air Setpoint Reduction', ylabel = 'Hour', title = 'Maximum Reduction of Cost by Air Setpoint Reduction and Hour')
	fig = scatter_fig.get_figure()
	plt.savefig(save_loc + 'opt_results_heat_max.png', bbox_inches='tight', s = 20, dpi = 300, figsize = (12,12))
	plt.clf()
	files_created.append('opt_results_heat_max.png')

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