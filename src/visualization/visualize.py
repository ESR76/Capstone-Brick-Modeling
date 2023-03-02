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
		print("in visualize..")
		direc = params['temp_output']
	else:
		print("in run -> visualize")
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

	### VISUALIZATION of OPTIMIZATION RESULTS ###

	# TO DO

	return "Visualize complete."