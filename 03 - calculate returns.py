import pandas as pd
import numpy as np
import yfinance as yf

# Download Adjusted Close prices for Apple between 2000 and 2010
price_data = yf.download('AAPL', start='2000-01-01', end='2010-12-31', progress=False)

# Keep only 'Adj Close' column and rename it
price_data = price_data.loc[:, ['Adj Close']]
price_data.rename(columns={'Adj Close': 'adj_close'}, inplace=True)

# Calculate simple returns (percentage change)
price_data['simple_rtn'] = price_data['adj_close'].pct_change()

# Calculate log returns - natural log of returns ratio
price_data['log_rtn'] = np.log(price_data['adj_close'] / price_data['adj_close'].shift(1))

# Drop rows with NaN (first row where returns can't be calculated)
price_data.dropna(inplace=True)

# Optional: Calculate annualized return and volatility
trading_days = 252
annualized_return = (1 + price_data['simple_rtn'].mean()) ** trading_days - 1
annualized_volatility = price_data['simple_rtn'].std() * np.sqrt(trading_days)

print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_volatility:.2%}")
