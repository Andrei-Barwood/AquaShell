import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def realized_volatility(x):
    # x expected to be returns within one month
    return np.sqrt(np.sum(x**2))

# Ensure df.index is datetime for grouping
df.index = pd.to_datetime(df.index)

# Calculate realized volatility per month
df_rv = df['log_rtn'].groupby(pd.Grouper(freq='M')).apply(realized_volatility)
df_rv = df_rv.to_frame(name='rv')

# Annualize monthly realized volatility by sqrt(12)
df_rv['rv'] = df_rv['rv'] * np.sqrt(12)

# Plot
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10,6))
ax[0].plot(df.index, df['log_rtn'])
ax[0].set_title('Log Returns')
ax[0].grid(True)

ax[1].plot(df_rv.index, df_rv['rv'], color='orange')
ax[1].set_title('Annualized Realized Volatility')
ax[1].grid(True)

plt.tight_layout()
plt.show()
