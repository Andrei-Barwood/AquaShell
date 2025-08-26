# calculate the rolling mean and standard deviation:
df_rolling = df[['simple_rtn']].rolling(window=21).agg(['mean', 'std'])
df_rolling.columns = df_rolling.columns.droplevel()

# join the rolling metrics to the original data:
df_outliers = df.join(df_rolling)

# define a function for detecting outliers:
def indentify_outliers(row, n_sigmas=3):
	x = row['simple_rtn']
	mu = row['mean']
	sigma = row['std']
	if (x > mu + 3 * sigma) | (x < mu - 3 * sigma):
		return 1
	else:
		return 0

# identify the outliers and extract their values for later use:
df_outliers['outlier'] = df_outliers.apply(indentify_outliers, axis=1)
outliers = df_outliers.loc[df_outliers['outlier']== 1, ['simple_rtn']]

# plot the results
fig, ax = plt.subplots()
ax.plot(df_outliers.index, df_outliers.simple_rtn, color='blue', label='Normal')
ax.scatter(outliers.index, outliers.simple_rtn, color='red', label='Anomaly')
ax.set_title("Apple's stock returns")
ax.legend(loc='lower right')