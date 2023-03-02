import pandas as pd

### UNUSED - ORIGINALLY HAD OPTIMIZE PIPELINE SEPARATE - IT DOESN'T MAKE SENSE TO RUN SEPARATE ###
# Would be called from clean_raw in clean_features
def optimize_cleaning(data, **params):
	data.loc[:, 'hour'] = data.loc[:, params['time_changed']].transform(lambda x: x.hour)
	medians = data.groupby(params['time_changed']).median()
	hour_medians = data.groupby(['hour']).median()

	min_ts = medians.index[0]
	max_ts = medians.index[len(medians) - 1]

	missingtimes_df = pd.DataFrame(index = pd.date_range(min_ts, max_ts, freq=params['time_floor_val']))

	complete_times = missingtimes_df.merge(medians, left_index = True, right_index = True, how = 'outer')
	complete_times.loc[:, 'hour'] = complete_times.index.hour

	keep_cols = list(complete_times.columns)[0: len(complete_times.columns) - 1]
	keep_cols_y = [x + "_y" for x in keep_cols]

	# merge then keep the relevant columns based on merge logic (no the most efficient but didn't have time to clean)
	imputed_meds = complete_times.loc[complete_times[params['energy_col']].isna(), :].merge(hour_medians, left_on = 'hour', right_index = True)
	imputed_meds = imputed_meds.loc[:, keep_cols_y].rename(dict(zip(keep_cols_y, keep_cols)), axis = 1)

	complete_times.loc[(complete_times[params['energy_col']].isna()), :] = imputed_meds

	return complete_times.reset_index().rename({'index': params['time_col']}, axis = 1).drop(['hour'], axis = 1)