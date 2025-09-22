from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from .portfolio import Portfolio
from .strategy_base import BaseStrategy, StrategySignal, Signal
from .data_manager import DataManager


class Backtester:
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission_per_share: float = 0.005,
                 slippage_bps: float = 2.0):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital
            commission_per_share: Commission per share traded
            slippage_bps: Slippage in basis points (100 bps = 1%)
        """
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps
        
        self.portfolio = Portfolio(initial_capital, commission_per_share)
        self.data_manager = DataManager()
        
        # Backtest state
        self.strategies: List[BaseStrategy] = []
        self.symbols: List[str] = []
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_datetime: Optional[datetime] = None
        
        # Results tracking
        self.signals_log: List[Dict] = []
        self.execution_log: List[Dict] = []
        
    def add_strategy(self, strategy: BaseStrategy):
        """Add a trading strategy to the backtester."""
        self.strategies.append(strategy)
        
    def set_data(self, historical_data: Dict[str, pd.DataFrame]):
        """
        Set historical data for backtesting.
        
        Args:
            historical_data: Dict mapping symbols to their OHLCV DataFrames
        """
        self.historical_data = historical_data
        self.symbols = list(historical_data.keys())
        self.data_manager.data_cache = historical_data
        
        # Initialize strategies with data
        for strategy in self.strategies:
            strategy.initialize(self.symbols, self.historical_data)
    
    async def load_data(self, 
                       symbols: List[str],
                       duration: str = "30 D",
                       bar_size: str = "1 min",
                       delay_between_requests: float = 1.0):
        """
        Load historical data from IBKR.
        
        Args:
            symbols: List of ticker symbols
            duration: Duration string
            bar_size: Bar size
            delay_between_requests: Delay between API requests in seconds
        """
        print(f"Loading data for symbols: {symbols}")
        data = await self.data_manager.fetch_historical_data(
            symbols, duration, bar_size, delay_between_requests=delay_between_requests
        )
        
        if not data:
            raise ValueError("No data was successfully fetched from IBKR")
        
        self.set_data(data)
        print(f"Loaded data for {len(data)} symbols")
    
    def run_backtest(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> Dict:
        """
        Run the backtest simulation.
        
        Args:
            start_date: Start date for backtest (default: earliest data)
            end_date: End date for backtest (default: latest data)
            
        Returns:
            Dictionary with backtest results
        """
        if not self.strategies:
            raise ValueError("No strategies added to backtester")
        
        if not self.historical_data:
            raise ValueError("No historical data loaded")
        
        print(f"Running backtest with {len(self.strategies)} strategies on {len(self.symbols)} symbols")
        
        # Determine backtest date range
        all_timestamps = []
        for df in self.historical_data.values():
            all_timestamps.extend(df.index.tolist())
        
        all_timestamps = sorted(set(all_timestamps))
        
        if start_date is None:
            start_date = all_timestamps[0]
        if end_date is None:
            end_date = all_timestamps[-1]
        
        # Filter timestamps to backtest range
        backtest_timestamps = [ts for ts in all_timestamps if start_date <= ts <= end_date]
        
        print(f"Backtesting from {start_date} to {end_date} ({len(backtest_timestamps)} bars)")
        
        # Run simulation
        for i, timestamp in enumerate(backtest_timestamps):
            self.current_datetime = timestamp
            self._process_bar(timestamp)
            
            # Progress update
            if i % 1000 == 0 or i == len(backtest_timestamps) - 1:
                progress = (i + 1) / len(backtest_timestamps) * 100
                print(f"Progress: {progress:.1f}% ({i + 1}/{len(backtest_timestamps)})")
        
        # Generate results
        results = self._generate_results(start_date, end_date)
        
        print(f"Backtest completed. Final equity: ${self.portfolio.total_equity:,.2f}")
        print(f"Total return: {((self.portfolio.total_equity / self.initial_capital) - 1) * 100:.2f}%")
        print(f"Total trades: {len(self.portfolio.trades)}")
        
        return results
    
    def _process_bar(self, timestamp: datetime):
        """Process a single bar across all symbols."""
        # Get current bar data for all symbols
        current_bars = {}
        historical_data_up_to_now = {}
        
        for symbol in self.symbols:
            df = self.historical_data[symbol]
            
            # Get data up to current timestamp
            data_up_to_now = df[df.index <= timestamp]
            if data_up_to_now.empty:
                continue
                
            current_bar = data_up_to_now.iloc[-1]
            current_bars[symbol] = current_bar
            historical_data_up_to_now[symbol] = data_up_to_now
        
        if not current_bars:
            return
        
        # Update portfolio with current prices
        current_prices = {symbol: bar['close'] for symbol, bar in current_bars.items()}
        self.portfolio.update_prices(current_prices, timestamp)
        
        # Generate signals from all strategies
        all_signals = []
        for strategy in self.strategies:
            try:
                # For each symbol, call the strategy with proper symbol context
                for symbol, current_bar in current_bars.items():
                    entry_signals = strategy.generate_signals(
                        current_bar, 
                        historical_data_up_to_now[symbol], 
                        self.portfolio, 
                        timestamp,
                        symbol=symbol  # Pass symbol explicitly
                    )
                    for signal in entry_signals:
                        signal.metadata['strategy'] = strategy.name
                        all_signals.append(signal)
                
                # Check exit conditions for existing positions
                for symbol in list(self.portfolio.positions.keys()):
                    if symbol in current_bars:
                        exit_signal = strategy.should_exit(
                            symbol, 
                            current_bars[symbol], 
                            historical_data_up_to_now[symbol],
                            self.portfolio, 
                            timestamp
                        )
                        if exit_signal:
                            exit_signal.metadata['strategy'] = strategy.name
                            all_signals.append(exit_signal)
                            
            except Exception as e:
                print(f"Error in strategy {strategy.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Log all signals
        for signal in all_signals:
            self.signals_log.append({
                'timestamp': timestamp,
                'strategy': signal.metadata.get('strategy'),
                'symbol': signal.symbol,
                'signal': signal.signal,
                'quantity': signal.quantity,
                'price': signal.price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            })
        
        # Execute signals
        self._execute_signals(all_signals, current_bars, timestamp)
    
    def _execute_signals(self, 
                        signals: List[StrategySignal], 
                        current_bars: Dict[str, pd.Series],
                        timestamp: datetime):
        """Execute trading signals."""
        for signal in signals:
            if signal.symbol not in current_bars:
                continue
                
            current_bar = current_bars[signal.symbol]
            execution_price = signal.price if signal.price else current_bar['close']
            
            # Apply slippage
            if signal.signal in [Signal.BUY]:
                execution_price *= (1 + self.slippage_bps / 10000)
            elif signal.signal in [Signal.SELL]:
                execution_price *= (1 - self.slippage_bps / 10000)
            
            # Execute the trade
            success = False
            if signal.signal == Signal.BUY and signal.quantity > 0:
                success = self.portfolio.open_position(
                    signal.symbol, signal.quantity, execution_price, timestamp
                )
            elif signal.signal == Signal.SELL and signal.quantity > 0:
                success = self.portfolio.open_position(
                    signal.symbol, -signal.quantity, execution_price, timestamp
                )
            elif signal.signal == Signal.CLOSE:
                success = self.portfolio.close_position(
                    signal.symbol, execution_price, timestamp
                )
            
            # Log execution
            self.execution_log.append({
                'timestamp': timestamp,
                'strategy': signal.metadata.get('strategy'),
                'symbol': signal.symbol,
                'signal': signal.signal,
                'quantity': signal.quantity,
                'requested_price': signal.price,
                'execution_price': execution_price,
                'success': success
            })
    
    def _generate_results(self, start_date: datetime, end_date: datetime) -> Dict:
        """Generate comprehensive backtest results."""
        equity_df = self.portfolio.get_equity_curve_df()
        trades_df = self.portfolio.get_trades_df()
        
        results = {
            'summary': {
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': self.initial_capital,
                'final_equity': self.portfolio.total_equity,
                'total_return': ((self.portfolio.total_equity / self.initial_capital) - 1) * 100,
                'total_trades': len(self.portfolio.trades),
                'winning_trades': len([t for t in self.portfolio.trades if t.is_winner]),
                'losing_trades': len([t for t in self.portfolio.trades if not t.is_winner]),
                'win_rate': len([t for t in self.portfolio.trades if t.is_winner]) / len(self.portfolio.trades) * 100 if self.portfolio.trades else 0,
            },
            'portfolio': self.portfolio,
            'equity_curve': equity_df,
            'trades': trades_df,
            'signals_log': pd.DataFrame(self.signals_log),
            'execution_log': pd.DataFrame(self.execution_log),
            'strategies': [str(strategy) for strategy in self.strategies]
        }
        
        # Calculate additional metrics if we have trades
        if not trades_df.empty:
            results['summary'].update({
                'avg_win': trades_df[trades_df['is_winner']]['pnl'].mean() if any(trades_df['is_winner']) else 0,
                'avg_loss': trades_df[~trades_df['is_winner']]['pnl'].mean() if any(~trades_df['is_winner']) else 0,
                'largest_win': trades_df['pnl'].max(),
                'largest_loss': trades_df['pnl'].min(),
                'gross_profit': trades_df[trades_df['pnl'] > 0]['pnl'].sum(),
                'gross_loss': trades_df[trades_df['pnl'] < 0]['pnl'].sum(),
            })
            
            if results['summary']['avg_loss'] != 0:
                results['summary']['profit_factor'] = abs(results['summary']['gross_profit'] / results['summary']['gross_loss'])
        
        # Calculate drawdown if we have equity curve
        if not equity_df.empty:
            equity_df['peak'] = equity_df['equity'].expanding().max()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
            results['summary']['max_drawdown'] = equity_df['drawdown'].min()
        
        return results