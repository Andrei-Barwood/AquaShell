import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Corrected rolling windows and improved calculations
df['moving_std_252'] = df['log_rtn'].rolling(window=252).std()  # Fixed: was 32, should be 252
df['moving_std_21'] = df['log_rtn'].rolling(window=21).std()

# Optional: Annualize the volatility for better interpretation
df['annualized_vol_252'] = df['moving_std_252'] * np.sqrt(252)
df['annualized_vol_21'] = df['moving_std_21'] * np.sqrt(252)

# Improved plotting
fig, ax = plt.subplots(3, 1, figsize=(18, 15), sharex=True)

# Stock price plot
df['adj_close'].plot(ax=ax[0])  # Removed unnecessary double brackets
ax[0].set_title('AAPL Time Series')
ax[0].set_ylabel('Stock Price ($)')
ax[0].grid(True, alpha=0.3)

# Log returns plot
df['log_rtn'].plot(ax=ax[1])
ax[1].set_ylabel('Log Returns')
ax[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)  # Add zero line
ax[1].grid(True, alpha=0.3)

# Volatility plot
df['moving_std_252'].plot(ax=ax[2], color='r', label='Moving Volatility 252d', linewidth=2)
df['moving_std_21'].plot(ax=ax[2], color='g', label='Moving Volatility 21d', alpha=0.7)

ax[2].set_ylabel('Moving Volatility')
ax[2].set_xlabel('Date')
ax[2].legend()
ax[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
