import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib.pyplot as plt  
import statsmodels.tsa.api as smt

# Assume df is defined, e.g., load MSFT data and compute log returns:
# Example:
# data = yf.download("MSFT", start="2020-01-01", end="2023-01-01")
# data['log_rtn'] = np.log(data['Adj Close']).diff()
# df = data.dropna(subset=['log_rtn'])

# Calculate the probability density function based on log returns
r_range = np.linspace(df.log_rtn.min(), df.log_rtn.max(), num=1000)
mu = df.log_rtn.mean()
sigma = df.log_rtn.std()
norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)  # Fixed variable m->mu

# Plot histogram and Q-Q plot side by side
fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # Fixed figsize tuple syntax

# Histogram of log returns with normal density overlay
sns.histplot(df.log_rtn, kde=False, stat='density', ax=ax[0])  # Updated from deprecated distplot
ax[0].plot(r_range, norm_pdf, 'g', lw=2, label=f'N({mu:.2f}, {sigma**2:.4f})')
ax[0].set_title('Distribution of MSFT Log Returns', fontsize=16)
ax[0].set_xlabel('Log Return')
ax[0].set_ylabel('Density')
ax[0].legend(loc='upper left')
ax[0].grid(True)

# Q-Q plot against normal distribution
sm.qqplot(df.log_rtn.values, line='s', ax=ax[1])
ax[1].set_title('Q-Q Plot of MSFT Log Returns', fontsize=16)
ax[1].grid(True)

plt.tight_layout()
plt.show()
