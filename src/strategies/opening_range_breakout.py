from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Add parent directory to path to import backtester modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtester.strategy_base import BaseStrategy, StrategySignal, Signal
from backtester.portfolio import Portfolio


class OpeningRangeBreakout(BaseStrategy):
    def __init__(self, 
                 opening_range_minutes: int = 30,
                 min_range_percent: float = 0.5,
                 max_range_percent: float = 5.0,
                 position_size_percent: float = 2.0,
                 stop_loss_percent: float = 2.0,
                 profit_target_percent: float = 4.0,
                 max_positions: int = 3,
                 market_open_hour: int = 9,
                 market_open_minute: int = 30):
        """
        Opening Range Breakout Strategy.
        
        Args:
            opening_range_minutes: Duration of opening range in minutes (default: 30)
            min_range_percent: Minimum range as % of stock price to consider valid (default: 0.5%)
            max_range_percent: Maximum range as % of stock price to consider valid (default: 5.0%)
            position_size_percent: Position size as % of portfolio equity (default: 2.0%)
            stop_loss_percent: Stop loss as % below/above entry (default: 2.0%)
            profit_target_percent: Profit target as % above/below entry (default: 4.0%)
            max_positions: Maximum concurrent positions (default: 3)
            market_open_hour: Market open hour (default: 9)
            market_open_minute: Market open minute (default: 30)
        """
        super().__init__(
            name="OpeningRangeBreakout",
            opening_range_minutes=opening_range_minutes,
            min_range_percent=min_range_percent,
            max_range_percent=max_range_percent,
            position_size_percent=position_size_percent,
            stop_loss_percent=stop_loss_percent,
            profit_target_percent=profit_target_percent,
            max_positions=max_positions,
            market_open_hour=market_open_hour,
            market_open_minute=market_open_minute
        )
        
        # Initialize state for tracking opening ranges
        self.state['opening_ranges'] = {}  # symbol -> {date: range_data}
        self.state['active_setups'] = {}   # symbol -> setup_data
        self.state['position_entry_prices'] = {}  # symbol -> entry_price
    
    def initialize(self, symbols: List[str], historical_data: Dict[str, pd.DataFrame]):
        """Initialize strategy with historical data."""
        print(f"Initializing ORB strategy for {len(symbols)} symbols")
        
        # Pre-calculate opening ranges for all trading days
        for symbol in symbols:
            if symbol not in historical_data:
                continue
                
            df = historical_data[symbol]
            self._calculate_all_opening_ranges(symbol, df)
        
        print(f"ORB strategy initialized with opening ranges for {len(self.state['opening_ranges'])} symbols")
    
    def _calculate_all_opening_ranges(self, symbol: str, df: pd.DataFrame):
        """Pre-calculate opening ranges for all trading days."""
        if symbol not in self.state['opening_ranges']:
            self.state['opening_ranges'][symbol] = {}
        
        # Group by trading date
        df_copy = df.copy()
        df_copy['date'] = df_copy.index.date
        
        for date in df_copy['date'].unique():
            daily_data = df_copy[df_copy['date'] == date]
            
            # Find market open time for this date - ensure timezone consistency
            market_open = datetime.combine(date, datetime.min.time())
            market_open = market_open.replace(
                hour=self.get_parameter('market_open_hour'),
                minute=self.get_parameter('market_open_minute')
            )
            
            range_end = market_open + timedelta(minutes=self.get_parameter('opening_range_minutes'))
            
            # Convert to pandas Timestamp for comparison with DataFrame index
            market_open_ts = pd.Timestamp(market_open)
            range_end_ts = pd.Timestamp(range_end)
            
            # Get opening range data - find first bar at or after market open
            # This handles cases where exact 9:30 AM data might not exist
            market_data_start = daily_data[daily_data.index >= market_open_ts]
            if market_data_start.empty:
                continue  # No data for this day after market open
            
            actual_start_time = market_data_start.index[0]
            actual_end_time = actual_start_time + timedelta(minutes=self.get_parameter('opening_range_minutes'))
            
            opening_range_data = daily_data[
                (daily_data.index >= actual_start_time) & 
                (daily_data.index < actual_end_time)
            ]
            
            if not opening_range_data.empty:
                or_high = opening_range_data['high'].max()
                or_low = opening_range_data['low'].min()
                or_range = or_high - or_low
                
                # Calculate range as percentage of stock price
                avg_price = (or_high + or_low) / 2
                range_percent = (or_range / avg_price) * 100 if avg_price > 0 else 0
                
                # Only store if range meets criteria
                min_range = self.get_parameter('min_range_percent')
                max_range = self.get_parameter('max_range_percent')
                
                if min_range <= range_percent <= max_range:
                    self.state['opening_ranges'][symbol][date] = {
                        'high': or_high,
                        'low': or_low,
                        'range': or_range,
                        'range_percent': range_percent,
                        'start_time': actual_start_time,
                        'end_time': actual_end_time,
                        'avg_price': avg_price
                    }
    
    def generate_signals(self, 
                        current_bar: pd.Series, 
                        historical_data: pd.DataFrame, 
                        portfolio: Portfolio,
                        timestamp: datetime,
                        symbol: str = None) -> List[StrategySignal]:
        """Generate entry signals for opening range breakouts."""
        signals = []
        
        # Use explicitly passed symbol or fallback to bar name
        if symbol is None:
            symbol = current_bar.name if hasattr(current_bar, 'name') else 'UNKNOWN'
        
        current_date = timestamp.date()
        current_price = current_bar['close']
        
        # Debug logging
        debug_info = {
            'symbol': symbol,
            'timestamp': timestamp,
            'current_price': current_price,
            'has_max_positions': len(portfolio.positions) >= self.get_parameter('max_positions'),
            'has_existing_position': not portfolio.is_flat(symbol),
            'has_opening_range': symbol in self.state['opening_ranges'] and current_date in self.state['opening_ranges'][symbol]
        }
        
        # Check if we already have max positions
        if len(portfolio.positions) >= self.get_parameter('max_positions'):
            return signals
        
        # Check if we already have a position in this symbol
        if not portfolio.is_flat(symbol):
            return signals
        
        # Check if we have an opening range for today
        if (symbol not in self.state['opening_ranges'] or 
            current_date not in self.state['opening_ranges'][symbol]):
            debug_info['no_opening_range_reason'] = f"Symbol {symbol} not in ranges or date {current_date} not found"
            if self.get_state('debug_mode', False):
                print(f"[ORB DEBUG] {timestamp} {symbol}: No opening range - {debug_info['no_opening_range_reason']}")
            return signals
        
        opening_range = self.state['opening_ranges'][symbol][current_date]
        debug_info['opening_range'] = opening_range
        
        # Only trade after opening range period has ended
        if timestamp < opening_range['end_time']:
            debug_info['before_range_end'] = True
            if self.get_state('debug_mode', False):
                print(f"[ORB DEBUG] {timestamp} {symbol}: Before range end time {opening_range['end_time']}")
            return signals
        
        # Check for breakout conditions
        or_high = opening_range['high']
        or_low = opening_range['low']
        
        # Calculate position size
        position_size = self._calculate_position_size(portfolio, current_price)
        debug_info.update({
            'or_high': or_high,
            'or_low': or_low,
            'position_size': position_size,
            'bullish_breakout': current_price > or_high,
            'bearish_breakout': current_price < or_low
        })
        
        if position_size <= 0:
            debug_info['position_size_zero'] = True
            if self.get_state('debug_mode', False):
                print(f"[ORB DEBUG] {timestamp} {symbol}: Position size is 0 (portfolio equity: ${portfolio.total_equity:,.2f})")
            return signals
        
        # Log breakout analysis
        if self.get_state('debug_mode', False) and (current_price > or_high or current_price < or_low):
            print(f"[ORB DEBUG] {timestamp} {symbol}: BREAKOUT DETECTED! Price: ${current_price:.2f}, Range: ${or_low:.2f}-${or_high:.2f}")
        
        # Check for bullish breakout
        if current_price > or_high:
            # Calculate stop loss and profit target
            stop_loss = or_low
            profit_target = current_price + (current_price - or_low) * (
                self.get_parameter('profit_target_percent') / self.get_parameter('stop_loss_percent')
            )
            
            signals.append(StrategySignal(
                signal=Signal.BUY,
                symbol=symbol,
                quantity=position_size,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=profit_target,
                metadata={
                    'setup_type': 'bullish_breakout',
                    'opening_range_high': or_high,
                    'opening_range_low': or_low,
                    'breakout_price': current_price
                }
            ))
            
            # Store entry price for exit logic
            self.state['position_entry_prices'][symbol] = current_price
            
            if self.get_state('debug_mode', False):
                print(f"[ORB DEBUG] {timestamp} {symbol}: BUY SIGNAL GENERATED! Size: {position_size}, Stop: ${stop_loss:.2f}, Target: ${profit_target:.2f}")
        
        # Check for bearish breakout
        elif current_price < or_low:
            # Calculate stop loss and profit target
            stop_loss = or_high
            profit_target = current_price - (or_high - current_price) * (
                self.get_parameter('profit_target_percent') / self.get_parameter('stop_loss_percent')
            )
            
            signals.append(StrategySignal(
                signal=Signal.SELL,
                symbol=symbol,
                quantity=position_size,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=profit_target,
                metadata={
                    'setup_type': 'bearish_breakout',
                    'opening_range_high': or_high,
                    'opening_range_low': or_low,
                    'breakout_price': current_price
                }
            ))
            
            # Store entry price for exit logic
            self.state['position_entry_prices'][symbol] = current_price
            
            if self.get_state('debug_mode', False):
                print(f"[ORB DEBUG] {timestamp} {symbol}: SELL SIGNAL GENERATED! Size: {position_size}, Stop: ${stop_loss:.2f}, Target: ${profit_target:.2f}")
        
        # Store debug info for analysis
        if not hasattr(self.state, 'debug_log'):
            self.state['debug_log'] = []
        self.state['debug_log'].append(debug_info)
        
        return signals
    
    def should_exit(self, 
                   position_symbol: str,
                   current_bar: pd.Series,
                   historical_data: pd.DataFrame,
                   portfolio: Portfolio,
                   timestamp: datetime) -> Optional[StrategySignal]:
        """Check if existing positions should be exited."""
        position = portfolio.get_position(position_symbol)
        if not position:
            return None
        
        current_price = current_bar['close']
        entry_price = self.state['position_entry_prices'].get(position_symbol)
        
        if not entry_price:
            # If we don't have entry price stored, use position entry price
            entry_price = position.entry_price
        
        # Get today's opening range for stop loss reference
        current_date = timestamp.date()
        opening_range = None
        if (position_symbol in self.state['opening_ranges'] and 
            current_date in self.state['opening_ranges'][position_symbol]):
            opening_range = self.state['opening_ranges'][position_symbol][current_date]
        
        # Check exit conditions
        if position.is_long:
            # Long position exit logic
            
            # Stop loss: opening range low
            if opening_range and current_price <= opening_range['low']:
                self._cleanup_position_state(position_symbol)
                return StrategySignal(
                    signal=Signal.CLOSE,
                    symbol=position_symbol,
                    price=current_price,
                    metadata={'exit_reason': 'stop_loss_or_low'}
                )
            
            # Alternative stop loss if no opening range
            stop_loss_price = entry_price * (1 - self.get_parameter('stop_loss_percent') / 100)
            if current_price <= stop_loss_price:
                self._cleanup_position_state(position_symbol)
                return StrategySignal(
                    signal=Signal.CLOSE,
                    symbol=position_symbol,
                    price=current_price,
                    metadata={'exit_reason': 'stop_loss_percent'}
                )
            
            # Profit target
            profit_target_price = entry_price * (1 + self.get_parameter('profit_target_percent') / 100)
            if current_price >= profit_target_price:
                self._cleanup_position_state(position_symbol)
                return StrategySignal(
                    signal=Signal.CLOSE,
                    symbol=position_symbol,
                    price=current_price,
                    metadata={'exit_reason': 'profit_target'}
                )
        
        else:
            # Short position exit logic
            
            # Stop loss: opening range high
            if opening_range and current_price >= opening_range['high']:
                self._cleanup_position_state(position_symbol)
                return StrategySignal(
                    signal=Signal.CLOSE,
                    symbol=position_symbol,
                    price=current_price,
                    metadata={'exit_reason': 'stop_loss_or_high'}
                )
            
            # Alternative stop loss if no opening range
            stop_loss_price = entry_price * (1 + self.get_parameter('stop_loss_percent') / 100)
            if current_price >= stop_loss_price:
                self._cleanup_position_state(position_symbol)
                return StrategySignal(
                    signal=Signal.CLOSE,
                    symbol=position_symbol,
                    price=current_price,
                    metadata={'exit_reason': 'stop_loss_percent'}
                )
            
            # Profit target
            profit_target_price = entry_price * (1 - self.get_parameter('profit_target_percent') / 100)
            if current_price <= profit_target_price:
                self._cleanup_position_state(position_symbol)
                return StrategySignal(
                    signal=Signal.CLOSE,
                    symbol=position_symbol,
                    price=current_price,
                    metadata={'exit_reason': 'profit_target'}
                )
        
        # End of day exit (close all positions before market close)
        market_close_time = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)
        if timestamp >= market_close_time - timedelta(minutes=5):
            self._cleanup_position_state(position_symbol)
            return StrategySignal(
                signal=Signal.CLOSE,
                symbol=position_symbol,
                price=current_price,
                metadata={'exit_reason': 'end_of_day'}
            )
        
        return None
    
    def _calculate_position_size(self, portfolio: Portfolio, current_price: float) -> int:
        """Calculate position size based on portfolio equity and position size percentage."""
        position_value = portfolio.total_equity * (self.get_parameter('position_size_percent') / 100)
        position_size = int(position_value / current_price)
        return max(position_size, 0)
    
    def _cleanup_position_state(self, symbol: str):
        """Clean up strategy state when position is closed."""
        if symbol in self.state['position_entry_prices']:
            del self.state['position_entry_prices'][symbol]
        if symbol in self.state['active_setups']:
            del self.state['active_setups'][symbol]
    
    def get_opening_ranges_summary(self) -> Dict:
        """Get summary of calculated opening ranges."""
        summary = {}
        for symbol, ranges in self.state['opening_ranges'].items():
            if ranges:
                range_percents = [r['range_percent'] for r in ranges.values()]
                summary[symbol] = {
                    'total_days': len(ranges),
                    'avg_range_percent': sum(range_percents) / len(range_percents),
                    'min_range_percent': min(range_percents),
                    'max_range_percent': max(range_percents)
                }
        return summary
    
    def get_debug_statistics(self) -> Dict:
        """Get debug statistics from strategy execution."""
        if 'debug_log' not in self.state or not self.state['debug_log']:
            return {}
        
        debug_df = pd.DataFrame(self.state['debug_log'])
        
        stats = {
            'total_evaluations': len(debug_df),
            'unique_symbols': debug_df['symbol'].nunique() if 'symbol' in debug_df else 0,
            'no_opening_range': debug_df['no_opening_range_reason'].notna().sum() if 'no_opening_range_reason' in debug_df else 0,
            'before_range_end': debug_df.get('before_range_end', False).sum(),
            'position_size_zero': debug_df.get('position_size_zero', False).sum(),
            'bullish_breakouts': debug_df.get('bullish_breakout', False).sum(),
            'bearish_breakouts': debug_df.get('bearish_breakout', False).sum(),
            'max_positions_reached': debug_df.get('has_max_positions', False).sum(),
            'existing_position': debug_df.get('has_existing_position', False).sum()
        }
        
        stats['total_breakouts'] = stats['bullish_breakouts'] + stats['bearish_breakouts']
        
        return stats