from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from .portfolio import Portfolio


class Signal:
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class StrategySignal:
    def __init__(self, 
                 signal: str, 
                 symbol: str, 
                 quantity: int = 0, 
                 price: Optional[float] = None,
                 stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None,
                 metadata: Optional[Dict] = None):
        self.signal = signal
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.metadata = metadata or {}


class BaseStrategy(ABC):
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.parameters = kwargs
        self.state = {}  # Strategy-specific state storage
        
    @abstractmethod
    def generate_signals(self, 
                        current_bar: pd.Series, 
                        historical_data: pd.DataFrame, 
                        portfolio: Portfolio,
                        timestamp: datetime) -> List[StrategySignal]:
        """
        Generate trading signals based on current market data.
        
        Args:
            current_bar: Current bar data (OHLCV)
            historical_data: Historical price data up to current bar
            portfolio: Current portfolio state
            timestamp: Current timestamp
            
        Returns:
            List of StrategySignal objects
        """
        pass
    
    @abstractmethod
    def should_exit(self, 
                   position_symbol: str,
                   current_bar: pd.Series,
                   historical_data: pd.DataFrame,
                   portfolio: Portfolio,
                   timestamp: datetime) -> Optional[StrategySignal]:
        """
        Check if existing positions should be exited.
        
        Args:
            position_symbol: Symbol of the position to check
            current_bar: Current bar data
            historical_data: Historical price data
            portfolio: Current portfolio state
            timestamp: Current timestamp
            
        Returns:
            StrategySignal if position should be exited, None otherwise
        """
        pass
    
    def initialize(self, symbols: List[str], historical_data: Dict[str, pd.DataFrame]):
        """
        Initialize strategy with historical data.
        Override this method if strategy needs pre-calculation setup.
        
        Args:
            symbols: List of symbols being traded
            historical_data: Dict mapping symbols to their historical data
        """
        pass
    
    def on_bar(self, 
               current_bars: Dict[str, pd.Series], 
               historical_data: Dict[str, pd.DataFrame],
               portfolio: Portfolio,
               timestamp: datetime) -> List[StrategySignal]:
        """
        Called on each new bar. Coordinates signal generation and exit checks.
        
        Args:
            current_bars: Dict mapping symbols to current bar data
            historical_data: Dict mapping symbols to historical data
            portfolio: Current portfolio state
            timestamp: Current timestamp
            
        Returns:
            List of all signals generated
        """
        all_signals = []
        
        # Check exit conditions for existing positions
        for symbol in portfolio.positions.keys():
            if symbol in current_bars:
                exit_signal = self.should_exit(
                    symbol, 
                    current_bars[symbol], 
                    historical_data[symbol],
                    portfolio, 
                    timestamp
                )
                if exit_signal:
                    all_signals.append(exit_signal)
        
        # Generate new entry signals
        for symbol, current_bar in current_bars.items():
            entry_signals = self.generate_signals(
                current_bar, 
                historical_data[symbol], 
                portfolio, 
                timestamp
            )
            all_signals.extend(entry_signals)
        
        return all_signals
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get strategy state value."""
        return self.state.get(key, default)
    
    def set_state(self, key: str, value: Any):
        """Set strategy state value."""
        self.state[key] = value
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get strategy parameter value."""
        return self.parameters.get(key, default)
    
    def __str__(self):
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.parameters.items())})"