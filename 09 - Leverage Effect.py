import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Download data
df = yf.download(['^GSPC', '^VIX'],
                start='1985-01-01',
                end='2018-12-31',
                progress=False)

# Select adjusted close prices
df = df[['Adj Close']]  # Fixed: 'Adj Close' not 'Adj close'
df.columns = df.columns.droplevel(0)
df = df.rename(columns={'^GSPC': 'sp500', '^VIX': 'vix'})

# Calculate log returns
df['log_rtn'] = np.log(df['sp500'] / df['sp500'].shift(1))
df['vol_rtn'] = np.log(df['vix'] / df['vix'].shift(1))
df.dropna(how='any', axis=0, inplace=True)

# Calculate correlation coefficient
corr_coeff = df['log_rtn'].corr(df['vol_rtn'])

# Create scatter plot with regression line
ax = sns.regplot(x='log_rtn', y='vol_rtn', data=df, line_kws={'color': 'red'})  # Fixed: x='log_rtn'
ax.set(title=f'S&P 500 vs. VIX ($\\rho$ = {corr_coeff:.2f})', 
       ylabel='VIX log returns', 
       xlabel='S&P 500 log returns')

plt.show()
