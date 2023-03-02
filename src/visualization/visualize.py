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

	# making visualizations of original data amounts
	plot_df = pd.read_csv(cwd + direc + params['out_name'])

	plot_df.loc[:, params['time_col']] = plot_df.loc[:, params['time_col']].transform(pd.Timestamp)
	plot_df = plot_df.loc[~plot_df['imputed'], :]

	#print(plot_df.head(5))
	vis_columns = params['viz_columns']

	fig, axes = plt.subplots(nrows=3, ncols=2, dpi=120, figsize=(10,6))
	for i, ax in enumerate(axes.flatten()):
	    data = plot_df.loc[:, [vis_columns[0], vis_columns[i + 1]]]
	    ax.plot(data.loc[:, vis_columns[0]], data.loc[:, vis_columns[i + 1]], color='red')
	    # Decorations
	    ax.set_title(vis_columns[i + 1])
	    ax.xaxis.set_ticks_position('none')
	    ax.yaxis.set_ticks_position('none')
	    ax.spines["top"].set_alpha(0)
	    ax.tick_params(labelsize=6)

	plt.tight_layout()

	plt.savefig(loc + 'data_values_pre_imputation.png')

	return "Visualize complete."