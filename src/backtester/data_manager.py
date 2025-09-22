import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from ib_async import IB
from ib_async.contract import Stock
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ibkr_connection import get_ibkr_connection_async


class DataManager:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
    async def fetch_historical_data(self, 
                                  symbols: List[str],
                                  duration: str = "30 D",
                                  bar_size: str = "1 min",
                                  what_to_show: str = "TRADES",
                                  use_rth: bool = True,
                                  end_datetime: str = "",
                                  delay_between_requests: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols sequentially to avoid rate limits.
        
        Args:
            symbols: List of ticker symbols
            duration: Duration string (e.g., "30 D", "1 Y")
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour")
            what_to_show: Data type to fetch
            use_rth: Use regular trading hours only
            end_datetime: End date/time for data
            delay_between_requests: Delay in seconds between requests (default: 1.0)
            
        Returns:
            Dictionary mapping symbols to their historical data DataFrames
        """
        print(f"Fetching historical data for {len(symbols)} symbols (sequential)...")
        
        ib = await get_ibkr_connection_async()
        data = {}
        
        try:
            # Fetch data sequentially to avoid rate limits
            for i, symbol in enumerate(symbols):
                print(f"[{i+1}/{len(symbols)}] Fetching {symbol}...")
                
                try:
                    result = await self._fetch_single_symbol(ib, symbol, duration, bar_size, what_to_show, use_rth, end_datetime)
                    
                    if result is not None and len(result) > 0:
                        df = self._bars_to_dataframe(result)
                        data[symbol] = df
                        self.data_cache[symbol] = df
                        print(f"✓ Fetched {len(df)} bars for {symbol}")
                    else:
                        print(f"✗ No data received for {symbol}")
                        
                except Exception as e:
                    print(f"✗ Error fetching data for {symbol}: {e}")
                    continue
                
                # Add delay between requests to avoid rate limits
                if i < len(symbols) - 1:  # Don't delay after last symbol
                    print(f"   Waiting {delay_between_requests}s before next request...")
                    await asyncio.sleep(delay_between_requests)
                    
        finally:
            ib.disconnect()
        
        print(f"✓ Data fetching complete. Successfully loaded {len(data)} out of {len(symbols)} symbols.")
        return data
    
    async def _fetch_single_symbol(self, ib: IB, symbol: str, duration: str, 
                                 bar_size: str, what_to_show: str, use_rth: bool, 
                                 end_datetime: str):
        """Fetch data for a single symbol."""
        try:
            contract = Stock(symbol, "SMART", "USD")
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth
            )
            return bars
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def _bars_to_dataframe(self, bars) -> pd.DataFrame:
        """Convert IB bars to pandas DataFrame with timezone handling."""
        data = []
        for bar in bars:
            data.append({
                'timestamp': bar.date,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume)
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert timezone-aware timestamps to timezone-naive for consistent handling
        if df['timestamp'].dt.tz is not None:
            print(f"   Converting from timezone: {df['timestamp'].dt.tz}")
            df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
        
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def get_data_for_timeframe(self, 
                              symbol: str, 
                              start_time: datetime, 
                              end_time: datetime) -> pd.DataFrame:
        """
        Get data for a specific symbol and timeframe.
        
        Args:
            symbol: Ticker symbol
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            DataFrame with data for the specified timeframe
        """
        if symbol not in self.data_cache:
            raise ValueError(f"No data available for symbol {symbol}")
        
        df = self.data_cache[symbol]
        return df[(df.index >= start_time) & (df.index <= end_time)]
    
    def get_latest_data(self, symbol: str, num_bars: int = 1) -> pd.DataFrame:
        """Get the latest N bars for a symbol."""
        if symbol not in self.data_cache:
            raise ValueError(f"No data available for symbol {symbol}")
        
        return self.data_cache[symbol].tail(num_bars)
    
    def calculate_opening_range(self, 
                               symbol: str, 
                               date: datetime, 
                               duration_minutes: int = 30) -> Dict[str, float]:
        """
        Calculate opening range for a specific date.
        
        Args:
            symbol: Ticker symbol
            date: Trading date
            duration_minutes: Duration of opening range in minutes
            
        Returns:
            Dict with 'high', 'low', 'range' keys
        """
        if symbol not in self.data_cache:
            raise ValueError(f"No data available for symbol {symbol}")
        
        df = self.data_cache[symbol]
        
        # Find market open for the date (assuming 9:30 AM EST)
        market_open = date.replace(hour=9, minute=30, second=0, microsecond=0)
        range_end = market_open + timedelta(minutes=duration_minutes)
        
        # Get data for opening range period
        opening_range_data = df[(df.index >= market_open) & (df.index < range_end)]
        
        if opening_range_data.empty:
            raise ValueError(f"No data available for opening range on {date.date()}")
        
        opening_range_high = opening_range_data['high'].max()
        opening_range_low = opening_range_data['low'].min()
        opening_range = opening_range_high - opening_range_low
        
        return {
            'high': opening_range_high,
            'low': opening_range_low,
            'range': opening_range,
            'start_time': market_open,
            'end_time': range_end
        }
    
    def get_trading_dates(self, symbol: str) -> List[datetime]:
        """Get list of unique trading dates for a symbol."""
        if symbol not in self.data_cache:
            raise ValueError(f"No data available for symbol {symbol}")
        
        df = self.data_cache[symbol]
        return df.index.date.unique().tolist()
    
    def resample_data(self, 
                     symbol: str, 
                     timeframe: str = "5min") -> pd.DataFrame:
        """
        Resample data to different timeframe.
        
        Args:
            symbol: Ticker symbol
            timeframe: New timeframe (e.g., "5min", "15min", "1H")
            
        Returns:
            Resampled DataFrame
        """
        if symbol not in self.data_cache:
            raise ValueError(f"No data available for symbol {symbol}")
        
        df = self.data_cache[symbol]
        
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def save_cache(self, filepath: str):
        """Save cached data to file."""
        if self.data_cache:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.data_cache, f)
            print(f"Data cache saved to {filepath}")
    
    def load_cache(self, filepath: str):
        """Load cached data from file."""
        import pickle
        try:
            with open(filepath, 'rb') as f:
                self.data_cache = pickle.load(f)
            print(f"Data cache loaded from {filepath}")
        except FileNotFoundError:
            print(f"Cache file {filepath} not found")
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    def get_symbols(self) -> List[str]:
        """Get list of symbols in cache."""
        return list(self.data_cache.keys())
    
    def clear_cache(self):
        """Clear all cached data."""
        self.data_cache.clear()