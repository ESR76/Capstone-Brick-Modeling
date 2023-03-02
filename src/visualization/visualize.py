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