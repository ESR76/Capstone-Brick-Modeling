import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

# should've changed the data but this is next best option - changing for visual only
def visualize_hour(x):
	replace = x - 8
	if replace < 0:
		replace = 24 + replace
	return replace

def visualize_results(cwd, opt_results, is_train, **params):
	loc = cwd + params['save_viz']
	direc = ""

	if not os.path.isdir(loc):
		os.mkdir(loc)

	if is_train:
		direc = params['temp_output']
	else:
		print("\nin run -> visualize for test data")
		direc = params['test_directory']


	### VISUALIZATION OF TRENDS BY UNIT OF TIME
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

	fig, axes = plt.subplots(nrows=3, ncols=1, dpi=300, figsize=(18,6))

	sns.lineplot(data = hour_groups.reset_index(), x = 'hour', y = 'energy', ax = axes[0])
	axes[0].set_title('Hour Means for Energy')
	sns.lineplot(data = weekday_groups.reset_index(), x = 'weekday', y = 'energy', ax = axes[1], sort = False)
	axes[1].set_title('Weekday Means for Energy')
	sns.lineplot(data = month_groups.reset_index(), x = 'month', y = 'energy', ax = axes[2], sort = False)
	axes[2].set_title('Month Means for Energy')


	fig.suptitle('Energy Means by Different Time Groups')
	plt.tight_layout()
	plt.savefig(loc + 'energy_time_means.png')
	plt.clf()

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
	plt.savefig(loc + 'data_values_pre_imputation.png')
	plt.clf()

	## OTHER VISUALIZATIONS ##

	### VISUALIZATION of OPTIMIZATION RESULTS ###
	fig, axes = plt.subplots(nrows=2, ncols=2, dpi=300, figsize=(18,6))
	sns.histplot(opt_results.loc[:, 'mean_difference'], ax = axes[0, 0])
	axes[0, 0].set(title = 'Mean Differences Histogram')

	sns.histplot(opt_results.loc[:, 'median_difference'], ax = axes[0, 1])
	axes[0, 1].set(title = 'Median Differences Histogram')

	sns.histplot(opt_results.loc[:, 'min_difference'], ax = axes[1, 0])
	axes[1, 0].set(title = 'Min Differences Histogram')

	sns.histplot(opt_results.loc[:, 'max_difference'], ax = axes[1, 1])
	axes[1, 1].set(title = 'Max Differences Histogram')

	fig.suptitle('Histograms of Differences')
	plt.tight_layout()
	plt.savefig(loc + 'opt_results_differences.png', bbox_inches='tight')
	plt.clf()


	fig, axes = plt.subplots(nrows=2, ncols=4, dpi=300, figsize=(18,12))
	# air
	sns.scatterplot(data = opt_results, x = 'air_decrease', y = 'median_difference', hue = 'air_limited', size = 'prop_boundary', ax = axes[0, 0], alpha = 0.7)
	axes[0, 0].set(xlabel = 'Air Supply Decrease', ylabel = 'Median Difference', title = 'Median Difference with Air Decrease')

	sns.scatterplot(data = opt_results, x = 'air_decrease', y = 'mean_difference', hue = 'air_limited', size = 'prop_boundary', ax = axes[0, 1], alpha = 0.7)
	axes[0, 1].set(xlabel = 'Air Supply Decrease', ylabel = 'Mean Difference', title = 'Mean Difference with Air Decrease')

	sns.scatterplot(data = opt_results, x = 'air_decrease', y = 'min_difference', hue = 'air_limited', size = 'prop_boundary', ax = axes[0, 2], alpha = 0.7)
	axes[0, 2].set(xlabel = 'Air Supply Decrease', ylabel = 'Min Difference', title = 'Min Difference with Air Decrease')

	sns.scatterplot(data = opt_results, x = 'air_decrease', y = 'max_difference', hue = 'air_limited', size = 'prop_boundary', ax = axes[0, 3], alpha = 0.7)
	axes[0, 3].set(xlabel = 'Air Supply Decrease', ylabel = 'Max Difference', title = 'Max Difference with Air Decrease')


	# temp
	sns.scatterplot(data = opt_results, x = 'temp_decrease', y = 'median_difference', hue = 'air_limited', size = 'prop_boundary', ax = axes[1, 0], alpha = 0.7)
	axes[1, 0].set(xlabel = 'Temp Decrease', ylabel = 'Median Difference', title = 'Median Difference with Temp Decrease')

	sns.scatterplot(data = opt_results, x = 'temp_decrease', y = 'mean_difference', hue = 'air_limited', size = 'prop_boundary', ax = axes[1, 1], alpha = 0.7)
	axes[1, 1].set(xlabel = 'Temp Decrease', ylabel = 'Mean Difference', title = 'Mean Difference with Temp Decrease')

	sns.scatterplot(data = opt_results, x = 'temp_decrease', y = 'min_difference', hue = 'air_limited', size = 'prop_boundary', ax = axes[1, 2], alpha = 0.7)
	axes[1, 2].set(xlabel = 'Temp Decrease', ylabel = 'Min Difference', title = 'Min Difference with Temp Decrease')

	sns.scatterplot(data = opt_results, x = 'temp_decrease', y = 'max_difference', hue = 'air_limited',size = 'prop_boundary',  ax = axes[1, 3], alpha = 0.7)
	axes[1, 3].set(xlabel = 'Temp Decrease', ylabel = 'Max Difference', title = 'Max Difference with Temp Decrease')

	fig.suptitle('Optimization Results by Setpoint Decreasing')
	plt.tight_layout()
	plt.savefig(loc + 'opt_results_scatter.png', box_inches='tight')
	plt.clf()

	# trying to get more info about low occupancy hours - examine relationships with prop boundary and hours TO DO

	return "Visualize complete."