import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Union, Optional

class FinancialAnalyzer:
    """
    A comprehensive financial analysis tool for stock market data.
    """
    
    def __init__(self, tickers: Union[str, List[str]], start_date: str, end_date: str):
        """
        Initialize the FinancialAnalyzer with tickers and date range.
        
        Parameters:
        - tickers: Stock symbol(s) (str or list of str)
        - start_date: Start date in 'YYYY-MM-DD' format
        - end_date: End date in 'YYYY-MM-DD' format
        """
        self.tickers = [tickers] if isinstance(tickers, str) else tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.volatility = None
        self.indicators = None
        
    def download_data(self, save_to_csv: bool = False) -> pd.DataFrame:
        """
        Download historical stock data for all tickers.
        
        Parameters:
        - save_to_csv: Whether to save data to CSV file
        
        Returns:
        - DataFrame with historical stock data
        """
        try:
            # Validate date format
            datetime.strptime(self.start_date, '%Y-%m-%d')
            datetime.strptime(self.end_date, '%Y-%m-%d')
            
            print(f"Downloading data for {', '.join(self.tickers)} from {self.start_date} to {self.end_date}...")
            
            # Download data for all tickers
            self.data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True,
                prepost=True,
                group_by='ticker'
            )
            
            # Validate data
            if self.data.empty:
                raise ValueError("No data retrieved. Check ticker symbols or date range.")
                
            # Handle single ticker case
            if len(self.tickers) == 1:
                self.data.columns = self.data.columns.droplevel(0)
            
            # Save to CSV if requested
            if save_to_csv:
                filename = f"{'_'.join(self.tickers)}_{self.start_date}_{self.end_date}.csv"
                self.data.to_csv(filename)
                print(f"Data saved to {filename}")
                
            print("Download successful!")
            return self.data
        
        except ValueError as ve:
            print(f"Validation Error: {ve}")
            return None
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None
    
    def validate_data(self, fill_method: str = 'ffill') -> pd.DataFrame:
        """
        Validate and clean data by handling missing values.
        
        Parameters:
        - fill_method: Method to fill missing values ('ffill', 'bfill', 'interpolate', 'drop')
        
        Returns:
        - Cleaned DataFrame
        """
        if self.data is None:
            print("No data available. Download data first.")
            return None
            
        print("Validating data and handling missing values...")
        
        # Check for missing values
        missing_before = self.data.isnull().sum().sum()
        print(f"Missing values before cleaning: {missing_before}")
        
        # Handle missing values based on method
        if fill_method == 'ffill':
            self.data = self.data.fillna(method='ffill')
        elif fill_method == 'bfill':
            self.data = self.data.fillna(method='bfill')
        elif fill_method == 'interpolate':
            self.data = self.data.interpolate()
        elif fill_method == 'drop':
            self.data = self.data.dropna()
        else:
            print("Invalid fill method. Using forward fill.")
            self.data = self.data.fillna(method='ffill')
        
        # Check remaining missing values
        missing_after = self.data.isnull().sum().sum()
        print(f"Missing values after cleaning: {missing_after}")
        
        return self.data
    
    def calculate_metrics(self) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Calculate financial metrics (returns, volatility, etc.).
        
        Returns:
        - Dictionary with calculated metrics
        """
        if self.data is None:
            print("No data available. Download data first.")
            return None
            
        print("Calculating financial metrics...")
        
        metrics = {}
        
        # Calculate daily returns
        if len(self.tickers) == 1:
            # Single ticker case
            self.returns = self.data['Close'].pct_change().dropna()
            metrics['daily_returns'] = self.returns
            
            # Calculate cumulative returns
            metrics['cumulative_returns'] = (1 + self.returns).cumprod() - 1
            
            # Calculate volatility (annualized standard deviation of daily returns)
            self.volatility = self.returns.std() * np.sqrt(252)
            metrics['volatility'] = self.volatility
            
            # Calculate Sharpe ratio (assuming risk-free rate of 2%)
            risk_free_rate = 0.02
            excess_returns = self.returns - risk_free_rate/252
            metrics['sharpe_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + self.returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            metrics['max_drawdown'] = drawdown.min()
            
        else:
            # Multiple tickers case
            returns_dict = {}
            volatility_dict = {}
            sharpe_dict = {}
            drawdown_dict = {}
            
            for ticker in self.tickers:
                # Calculate daily returns
                returns = self.data[ticker]['Close'].pct_change().dropna()
                returns_dict[ticker] = returns
                
                # Calculate cumulative returns
                metrics[f'{ticker}_cumulative_returns'] = (1 + returns).cumprod() - 1
                
                # Calculate volatility
                volatility = returns.std() * np.sqrt(252)
                volatility_dict[ticker] = volatility
                
                # Calculate Sharpe ratio
                risk_free_rate = 0.02
                excess_returns = returns - risk_free_rate/252
                sharpe_dict[ticker] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                
                # Calculate maximum drawdown
                cumulative_returns = (1 + returns).cumprod()
                peak = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - peak) / peak
                drawdown_dict[ticker] = drawdown.min()
            
            self.returns = pd.DataFrame(returns_dict)
            metrics['daily_returns'] = self.returns
            metrics['volatility'] = pd.Series(volatility_dict)
            metrics['sharpe_ratio'] = pd.Series(sharpe_dict)
            metrics['max_drawdown'] = pd.Series(drawdown_dict)
            
            # Calculate correlation matrix
            metrics['correlation'] = self.returns.corr()
        
        return metrics
    
    def calculate_indicators(self, 
                            ma_periods: List[int] = [20, 50],
                            rsi_period: int = 14,
                            bollinger_period: int = 20,
                            bollinger_std: int = 2) -> pd.DataFrame:
        """
        Calculate technical indicators (moving averages, RSI, Bollinger Bands, etc.).
        
        Parameters:
        - ma_periods: List of periods for moving averages
        - rsi_period: Period for RSI calculation
        - bollinger_period: Period for Bollinger Bands
        - bollinger_std: Number of standard deviations for Bollinger Bands
        
        Returns:
        - DataFrame with technical indicators
        """
        if self.data is None:
            print("No data available. Download data first.")
            return None
            
        print("Calculating technical indicators...")
        
        indicators = {}
        
        if len(self.tickers) == 1:
            # Single ticker case
            close_prices = self.data['Close']
            
            # Calculate moving averages
            for period in ma_periods:
                indicators[f'MA_{period}'] = close_prices.rolling(window=period).mean()
            
            # Calculate RSI
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            rs = avg_gain / avg_loss
            indicators['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands
            middle_band = close_prices.rolling(window=bollinger_period).mean()
            std_dev = close_prices.rolling(window=bollinger_period).std()
            indicators['BB_Upper'] = middle_band + (std_dev * bollinger_std)
            indicators['BB_Middle'] = middle_band
            indicators['BB_Lower'] = middle_band - (std_dev * bollinger_std)
            
            # Calculate MACD
            ema_12 = close_prices.ewm(span=12, adjust=False).mean()
            ema_26 = close_prices.ewm(span=26, adjust=False).mean()
            indicators['MACD'] = ema_12 - ema_26
            indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
            indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']
            
        else:
            # Multiple tickers case
            for ticker in self.tickers:
                close_prices = self.data[ticker]['Close']
                ticker_indicators = {}
                
                # Calculate moving averages
                for period in ma_periods:
                    ticker_indicators[f'MA_{period}'] = close_prices.rolling(window=period).mean()
                
                # Calculate RSI
                delta = close_prices.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=rsi_period).mean()
                avg_loss = loss.rolling(window=rsi_period).mean()
                rs = avg_gain / avg_loss
                ticker_indicators['RSI'] = 100 - (100 / (1 + rs))
                
                # Calculate Bollinger Bands
                middle_band = close_prices.rolling(window=bollinger_period).mean()
                std_dev = close_prices.rolling(window=bollinger_period).std()
                ticker_indicators['BB_Upper'] = middle_band + (std_dev * bollinger_std)
                ticker_indicators['BB_Middle'] = middle_band
                ticker_indicators['BB_Lower'] = middle_band - (std_dev * bollinger_std)
                
                # Calculate MACD
                ema_12 = close_prices.ewm(span=12, adjust=False).mean()
                ema_26 = close_prices.ewm(span=26, adjust=False).mean()
                ticker_indicators['MACD'] = ema_12 - ema_26
                ticker_indicators['MACD_Signal'] = ticker_indicators['MACD'].ewm(span=9, adjust=False).mean()
                ticker_indicators['MACD_Histogram'] = ticker_indicators['MACD'] - ticker_indicators['MACD_Signal']
                
                indicators[ticker] = pd.DataFrame(ticker_indicators)
        
        self.indicators = indicators
        return indicators
    
    def visualize(self, 
                  show_price: bool = True,
                  show_volume: bool = True,
                  show_returns: bool = True,
                  show_indicators: bool = True,
                  show_correlation: bool = True,
                  figsize: tuple = (15, 12)) -> None:
        """
        Visualize the data and analysis results.
        
        Parameters:
        - show_price: Whether to show price chart
        - show_volume: Whether to show volume chart
        - show_returns: Whether to show returns chart
        - show_indicators: Whether to show technical indicators
        - show_correlation: Whether to show correlation heatmap (for multiple tickers)
        - figsize: Figure size for plots
        """
        if self.data is None:
            print("No data available. Download data first.")
            return
            
        # Set style
        sns.set_style('darkgrid')
        plt.figure(figsize=figsize)
        
        if len(self.tickers) == 1:
            # Single ticker visualization
            ticker = self.tickers[0]
            n_rows = 0
            
            # Determine number of rows needed
            if show_price: n_rows += 1
            if show_volume: n_rows += 1
            if show_returns: n_rows += 1
            if show_indicators: n_rows += 3  # RSI, MACD, Bollinger Bands
            
            current_row = 1
            
            # Price chart
            if show_price:
                plt.subplot(n_rows, 1, current_row)
                plt.plot(self.data.index, self.data['Close'], label='Close Price')
                
                if self.indicators is not None and show_indicators:
                    # Add moving averages
                    for col in self.indicators.columns:
                        if col.startswith('MA_'):
                            plt.plot(self.indicators.index, self.indicators[col], label=col, alpha=0.7)
                    
                    # Add Bollinger Bands
                    if 'BB_Upper' in self.indicators.columns:
                        plt.plot(self.indicators.index, self.indicators['BB_Upper'], label='BB Upper', color='orange', alpha=0.5)
                        plt.plot(self.indicators.index, self.indicators['BB_Middle'], label='BB Middle', color='black', alpha=0.5)
                        plt.plot(self.indicators.index, self.indicators['BB_Lower'], label='BB Lower', color='orange', alpha=0.5)
                
                plt.title(f'{ticker} Price Chart')
                plt.ylabel('Price')
                plt.legend()
                current_row += 1
            
            # Volume chart
            if show_volume:
                plt.subplot(n_rows, 1, current_row)
                plt.bar(self.data.index, self.data['Volume'], color='blue', alpha=0.5)
                plt.title(f'{ticker} Volume')
                plt.ylabel('Volume')
                current_row += 1
            
            # Returns chart
            if show_returns and self.returns is not None:
                plt.subplot(n_rows, 1, current_row)
                plt.plot(self.returns.index, self.returns, label='Daily Returns')
                plt.axhline(y=0, color='r', linestyle='-')
                plt.title(f'{ticker} Daily Returns')
                plt.ylabel('Returns')
                plt.legend()
                current_row += 1
            
            # Technical indicators
            if show_indicators and self.indicators is not None:
                # RSI
                plt.subplot(n_rows, 1, current_row)
                plt.plot(self.indicators.index, self.indicators['RSI'], label='RSI')
                plt.axhline(y=70, color='r', linestyle='-')
                plt.axhline(y=30, color='g', linestyle='-')
                plt.title(f'{ticker} RSI')
                plt.ylabel('RSI')
                plt.legend()
                current_row += 1
                
                # MACD
                plt.subplot(n_rows, 1, current_row)
                plt.plot(self.indicators.index, self.indicators['MACD'], label='MACD')
                plt.plot(self.indicators.index, self.indicators['MACD_Signal'], label='Signal')
                plt.bar(self.indicators.index, self.indicators['MACD_Histogram'], label='Histogram', color='gray', alpha=0.5)
                plt.title(f'{ticker} MACD')
                plt.ylabel('MACD')
                plt.legend()
                current_row += 1
                
                # Bollinger Bands
                plt.subplot(n_rows, 1, current_row)
                plt.plot(self.data.index, self.data['Close'], label='Close Price')
                plt.plot(self.indicators.index, self.indicators['BB_Upper'], label='BB Upper', color='orange', alpha=0.5)
                plt.plot(self.indicators.index, self.indicators['BB_Middle'], label='BB Middle', color='black', alpha=0.5)
                plt.plot(self.indicators.index, self.indicators['BB_Lower'], label='BB Lower', color='orange', alpha=0.5)
                plt.title(f'{ticker} Bollinger Bands')
                plt.ylabel('Price')
                plt.legend()
                current_row += 1
            
            plt.tight_layout()
            plt.show()
            
        else:
            # Multiple tickers visualization
            if show_correlation and self.returns is not None:
                plt.figure(figsize=figsize)
                sns.heatmap(self.returns.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title('Correlation Matrix of Returns')
                plt.tight_layout()
                plt.show()
            
            # Price charts for all tickers
            if show_price:
                plt.figure(figsize=figsize)
                for ticker in self.tickers:
                    plt.plot(self.data[ticker]['Close'], label=ticker)
                plt.title('Price Comparison')
                plt.ylabel('Price')
                plt.legend()
                plt.tight_layout()
                plt.show()
            
            # Returns charts for all tickers
            if show_returns and self.returns is not None:
                plt.figure(figsize=figsize)
                for ticker in self.tickers:
                    plt.plot(self.returns[ticker], label=ticker)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.title('Daily Returns Comparison')
                plt.ylabel('Returns')
                plt.legend()
                plt.tight_layout()
                plt.show()
            
            # Technical indicators for each ticker
            if show_indicators and self.indicators is not None:
                for ticker in self.tickers:
                    plt.figure(figsize=figsize)
                    ticker_indicators = self.indicators[ticker]
                    
                    # Price and moving averages
                    plt.subplot(3, 1, 1)
                    plt.plot(self.data[ticker]['Close'], label='Close Price')
                    
                    # Add moving averages
                    for col in ticker_indicators.columns:
                        if col.startswith('MA_'):
                            plt.plot(ticker_indicators[col], label=col, alpha=0.7)
                    
                    # Add Bollinger Bands
                    if 'BB_Upper' in ticker_indicators.columns:
                        plt.plot(ticker_indicators['BB_Upper'], label='BB Upper', color='orange', alpha=0.5)
                        plt.plot(ticker_indicators['BB_Middle'], label='BB Middle', color='black', alpha=0.5)
                        plt.plot(ticker_indicators['BB_Lower'], label='BB Lower', color='orange', alpha=0.5)
                    
                    plt.title(f'{ticker} Price and Indicators')
                    plt.ylabel('Price')
                    plt.legend()
                    
                    # RSI
                    plt.subplot(3, 1, 2)
                    plt.plot(ticker_indicators['RSI'], label='RSI')
                    plt.axhline(y=70, color='r', linestyle='-')
                    plt.axhline(y=30, color='g', linestyle='-')
                    plt.title(f'{ticker} RSI')
                    plt.ylabel('RSI')
                    plt.legend()
                    
                    # MACD
                    plt.subplot(3, 1, 3)
                    plt.plot(ticker_indicators['MACD'], label='MACD')
                    plt.plot(ticker_indicators['MACD_Signal'], label='Signal')
                    plt.bar(ticker_indicators.index, ticker_indicators['MACD_Histogram'], label='Histogram', color='gray', alpha=0.5)
                    plt.title(f'{ticker} MACD')
                    plt.ylabel('MACD')
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.show()

# Example usage
if __name__ == "__main__":
    # Single ticker example
    print("=== Single Ticker Analysis ===")
    analyzer = FinancialAnalyzer('AAPL', '2000-01-01', '2010-12-31')
    analyzer.download_data(save_to_csv=True)
    analyzer.validate_data()
    metrics = analyzer.calculate_metrics()
    indicators = analyzer.calculate_indicators()
    
    # Print some metrics
    print("\nFinancial Metrics:")
    print(f"Volatility: {metrics['volatility']:.4f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
    
    # Visualize
    analyzer.visualize()
    
    # Multiple tickers example
    print("\n=== Multiple Tickers Analysis ===")
    analyzer_multi = FinancialAnalyzer(['AAPL', 'CSCO', 'GOOG'], '2015-01-01', '2020-12-31')
    analyzer_multi.download_data()
    analyzer_multi.validate_data()
    metrics_multi = analyzer_multi.calculate_metrics()
    indicators_multi = analyzer_multi.calculate_indicators()
    
    # Print some metrics
    print("\nFinancial Metrics:")
    print("Volatility:")
    print(metrics_multi['volatility'])
    print("\nSharpe Ratio:")
    print(metrics_multi['sharpe_ratio'])
    print("\nMax Drawdown:")
    print(metrics_multi['max_drawdown'])
    
    # Visualize
    analyzer_multi.visualize(show_correlation=True)