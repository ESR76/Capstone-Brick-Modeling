import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

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
	plot_df.loc[:, 'hour'] = plot_df.loc[:, params['time_col']].transform(lambda x: x.hour)
	plot_df.loc[:, 'weekday'] = plot_df.loc[:, params['time_col']].transform(lambda x: x.weekday)
	#plot_df.loc[:, 'weekday'] = plot_df.loc[:, 'weekday'].replace({0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thurs', 4: 'Fri', 5: 'Sat', 6: 'Sun'})
	plot_df.loc[:, 'month'] = plot_df.loc[:, params['time_col']].transform(lambda x: x.month)
	#plot_df.loc[:, 'month'] = plot_df.loc[:, 'month'].replace({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
	#plot_df.loc[:, 'year'] = plot_df.loc[:, params['time_col']].transform(lambda x: x.year)
	#plot_df.loc[:, 'month/year'] = plot_df.apply(lambda x: str(x['month']) + '/' + str(x['year']), axis = 1)

	# LOOK INTO THIS FOR LABELS: https://stackoverflow.com/questions/62630875/how-to-change-the-plot-order-of-the-categorical-x-axis

	#mo_year_groups = plot_df.groupby('month/year')['energy'].mean()
	month_groups = plot_df.groupby('month')['energy'].mean()
	weekday_groups = plot_df.groupby('weekday')['energy'].mean()#.replace({0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thurs', 4: 'Fri', 5: 'Sat', 6: 'Sun'})
	hour_groups = plot_df.groupby('hour')['energy'].mean()

	fig, axes = plt.subplots(nrows=3, ncols=1, dpi=120, figsize=(18,6))

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

	fig, axes = plt.subplots(nrows=2, ncols=3, dpi=120, figsize=(18,6))
	for i, ax in enumerate(axes.flatten()):
	    data = plot_df.loc[:, [vis_columns[0], vis_columns[i + 1]]]
	    ax.plot(data.loc[:, vis_columns[0]], data.loc[:, vis_columns[i + 1]], color='red')
	    # Decorations
	    ax.set_title(vis_columns[i + 1])
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
	current_fig = sns.histplot(opt_results.loc[:, 'mean_difference'])
	current_fig.set(title = 'Mean Differences Histogram')
	current_fig.figure.savefig(loc + 'opt_results_mean_differences.png', bbox_inches='tight')
	plt.clf()

	current_fig = sns.histplot(opt_results.loc[:, 'median_difference'])
	current_fig.set(title = 'Median Differences Histogram')
	current_fig.figure.savefig(loc + 'opt_results_median_differences.png', bbox_inches='tight')
	plt.clf()

	current_fig = sns.histplot(opt_results.loc[:, 'min_difference'])
	current_fig.set(title = 'Minimum Differences Histogram')
	current_fig.figure.savefig(loc + 'opt_results_min_differences.png', bbox_inches='tight')
	plt.clf()

	current_fig = sns.histplot(opt_results.loc[:, 'max_difference'])
	current_fig.set(title = 'Maximum Differences Histogram')
	current_fig.figure.savefig(loc + 'opt_results_max_differences.png', bbox_inches='tight')
	plt.clf()

	scatter_obj = sns.scatterplot(data = opt_results, x = 'temp_decrease', y = 'air_decrease', size = 'mean_difference')
	## LABELS - BBOX_INCHES keyword saves the labels
	scatter_obj.set(xlabel = 'Temperature Decrease', ylabel = 'Air Supply Decrease', title = 'Air Supply vs. Temperature Decrease')
	scatter_obj.figure.savefig(loc + 'opt_results_scatter_mean.png', bbox_inches='tight')
	plt.clf()

	# TO DO

	return "Visualize complete."