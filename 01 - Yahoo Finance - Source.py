import pandas as pd
import yfinance as yf
from datetime import datetime

def download_stock_data(ticker, start_date, end_date, save_to_csv=False):
    """
    Download historical stock data from Yahoo Finance.
    
    Parameters:
    - ticker: Stock symbol (str)
    - start_date: Start date in 'YYYY-MM-DD' format (str)
    - end_date: End date in 'YYYY-MM-DD' format (str)
    - save_to_csv: Whether to save data to CSV file (bool)
    
    Returns:
    - DataFrame with historical stock data
    """
    try:
        # Validate date format
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
        
        print(f"Downloading {ticker} data from {start_date} to {end_date}...")
        
        # Download data with error handling
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,  # Adjust for splits/dividends automatically
            prepost=True       # Include pre/post market data
        )
        
        # Validate data
        if df.empty:
            raise ValueError("No data retrieved. Check ticker symbol or date range.")
            
        # Basic data cleaning
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        df.index.name = 'date'
        
        # Save to CSV if requested
        if save_to_csv:
            filename = f"{ticker}_{start_date}_{end_date}.csv"
            df.to_csv(filename)
            print(f"Data saved to {filename}")
            
        print("Download successful!")
        return df
    
    except ValueError as ve:
        print(f"Validation Error: {ve}")
        return None
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Download Apple data
    aapl_data = download_stock_data(
        ticker='AAPL',
        start_date='2000-01-01',
        end_date='2010-12-31',
        save_to_csv=True
    )
    
    # Display first 5 rows if data exists
    if aapl_data is not None:
        print("\nSample data:")
        print(aapl_data.head())