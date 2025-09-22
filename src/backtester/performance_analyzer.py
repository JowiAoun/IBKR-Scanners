import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class PerformanceAnalyzer:
    def __init__(self, results: Dict):
        """
        Initialize performance analyzer with backtest results.
        
        Args:
            results: Dictionary containing backtest results from Backtester
        """
        self.results = results
        self.equity_curve = results.get('equity_curve', pd.DataFrame())
        self.trades = results.get('trades', pd.DataFrame())
        self.summary = results.get('summary', {})
        
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            
        Returns:
            Sharpe ratio
        """
        if self.equity_curve.empty:
            return 0.0
        
        # Calculate daily returns
        returns = self.equity_curve['equity'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # Convert to annual metrics
        daily_risk_free = risk_free_rate / 252
        excess_returns = returns - daily_risk_free
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return sharpe
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (focuses on downside deviation).
        
        Args:
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sortino ratio
        """
        if self.equity_curve.empty:
            return 0.0
        
        returns = self.equity_curve['equity'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        daily_risk_free = risk_free_rate / 252
        excess_returns = returns - daily_risk_free
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = downside_returns.std()
        
        if downside_deviation == 0:
            return 0.0
        
        sortino = (excess_returns.mean() / downside_deviation) * np.sqrt(252)
        return sortino
    
    def calculate_max_drawdown(self) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Returns:
            Dictionary with max drawdown, duration, and recovery info
        """
        if self.equity_curve.empty:
            return {'max_drawdown': 0.0, 'max_dd_duration': 0, 'recovery_time': 0}
        
        equity = self.equity_curve['equity']
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        
        max_drawdown = drawdown.min()
        
        # Find max drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = []
        start = None
        
        for i, in_dd in enumerate(is_drawdown):
            if in_dd and start is None:
                start = i
            elif not in_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        # Handle case where we end in drawdown
        if start is not None:
            drawdown_periods.append(len(is_drawdown) - start)
        
        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Calculate recovery time (time to new high after max drawdown)
        max_dd_idx = drawdown.idxmin()
        recovery_time = 0
        
        if max_dd_idx in equity.index:
            max_dd_equity = equity.loc[max_dd_idx]
            future_equity = equity.loc[max_dd_idx:]
            recovery_points = future_equity[future_equity > equity.loc[:max_dd_idx].max()]
            
            if not recovery_points.empty:
                recovery_idx = recovery_points.index[0]
                recovery_time = len(equity.loc[max_dd_idx:recovery_idx]) - 1
        
        return {
            'max_drawdown': max_drawdown,
            'max_dd_duration': max_dd_duration,
            'recovery_time': recovery_time
        }
    
    def calculate_win_rate_metrics(self) -> Dict[str, float]:
        """
        Calculate win rate and related trading metrics.
        
        Returns:
            Dictionary with win rate, profit factor, etc.
        """
        if self.trades.empty:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'expectancy': 0.0
            }
        
        winning_trades = self.trades[self.trades['pnl'] > 0]
        losing_trades = self.trades[self.trades['pnl'] < 0]
        
        total_trades = len(self.trades)
        num_winners = len(winning_trades)
        num_losers = len(losing_trades)
        
        win_rate = (num_winners / total_trades * 100) if total_trades > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        
        largest_win = winning_trades['pnl'].max() if not winning_trades.empty else 0
        largest_loss = losing_trades['pnl'].min() if not losing_trades.empty else 0
        
        # Expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)
        loss_rate = (num_losers / total_trades) if total_trades > 0 else 0
        expectancy = (avg_win * win_rate / 100) + (avg_loss * loss_rate)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'expectancy': expectancy,
            'total_trades': total_trades,
            'winning_trades': num_winners,
            'losing_trades': num_losers
        }
    
    def calculate_volatility(self) -> float:
        """Calculate annualized volatility of returns."""
        if self.equity_curve.empty:
            return 0.0
        
        returns = self.equity_curve['equity'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        return returns.std() * np.sqrt(252) * 100
    
    def calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (Annual return / Max drawdown)."""
        max_dd_info = self.calculate_max_drawdown()
        max_drawdown = abs(max_dd_info['max_drawdown'])
        
        if max_drawdown == 0:
            return float('inf')
        
        # Calculate annualized return
        if self.equity_curve.empty:
            return 0.0
        
        start_equity = self.equity_curve['equity'].iloc[0]
        end_equity = self.equity_curve['equity'].iloc[-1]
        total_return = (end_equity / start_equity - 1) * 100
        
        # Approximate annualized return (this is simplified)
        days = len(self.equity_curve)
        annual_return = total_return * (252 / days) if days > 0 else 0
        
        return annual_return / max_drawdown
    
    def analyze_trade_duration(self) -> Dict[str, float]:
        """Analyze trade duration statistics."""
        if self.trades.empty:
            return {
                'avg_duration_hours': 0.0,
                'min_duration_hours': 0.0,
                'max_duration_hours': 0.0,
                'median_duration_hours': 0.0
            }
        
        # Convert duration to hours
        durations_hours = self.trades['duration'].dt.total_seconds() / 3600
        
        return {
            'avg_duration_hours': durations_hours.mean(),
            'min_duration_hours': durations_hours.min(),
            'max_duration_hours': durations_hours.max(),
            'median_duration_hours': durations_hours.median()
        }
    
    def analyze_monthly_returns(self) -> pd.DataFrame:
        """Analyze monthly returns breakdown."""
        if self.equity_curve.empty:
            return pd.DataFrame()
        
        equity_df = self.equity_curve.copy()
        equity_df['year'] = equity_df['timestamp'].dt.year
        equity_df['month'] = equity_df['timestamp'].dt.month
        
        # Get monthly equity values (last day of each month)
        monthly_equity = equity_df.groupby(['year', 'month'])['equity'].last()
        monthly_returns = monthly_equity.pct_change() * 100
        
        return monthly_returns.reset_index()
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        report = {
            'basic_metrics': self.summary,
            'risk_metrics': {
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'sortino_ratio': self.calculate_sortino_ratio(),
                'calmar_ratio': self.calculate_calmar_ratio(),
                'volatility': self.calculate_volatility(),
            },
            'drawdown_analysis': self.calculate_max_drawdown(),
            'trade_analysis': self.calculate_win_rate_metrics(),
            'duration_analysis': self.analyze_trade_duration(),
        }
        
        # Add monthly returns if available
        monthly_returns = self.analyze_monthly_returns()
        if not monthly_returns.empty:
            report['monthly_returns'] = monthly_returns
        
        return report
    
    def print_performance_summary(self):
        """Print a formatted performance summary."""
        report = self.generate_performance_report()
        
        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*60)
        
        # Basic metrics
        basic = report['basic_metrics']
        print(f"\nBasic Metrics:")
        print(f"  Initial Capital: ${basic.get('initial_capital', 0):,.2f}")
        print(f"  Final Equity: ${basic.get('final_equity', 0):,.2f}")
        print(f"  Total Return: {basic.get('total_return', 0):.2f}%")
        print(f"  Total Trades: {basic.get('total_trades', 0)}")
        
        # Risk metrics
        risk = report['risk_metrics']
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {risk['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio: {risk['calmar_ratio']:.2f}")
        print(f"  Volatility: {risk['volatility']:.2f}%")
        
        # Drawdown
        dd = report['drawdown_analysis']
        print(f"\nDrawdown Analysis:")
        print(f"  Max Drawdown: {dd['max_drawdown']:.2f}%")
        print(f"  Max DD Duration: {dd['max_dd_duration']} periods")
        print(f"  Recovery Time: {dd['recovery_time']} periods")
        
        # Trade analysis
        trade = report['trade_analysis']
        print(f"\nTrade Analysis:")
        print(f"  Win Rate: {trade['win_rate']:.1f}%")
        print(f"  Profit Factor: {trade['profit_factor']:.2f}")
        print(f"  Average Win: ${trade['avg_win']:.2f}")
        print(f"  Average Loss: ${trade['avg_loss']:.2f}")
        print(f"  Expectancy: ${trade['expectancy']:.2f}")
        
        # Duration analysis
        duration = report['duration_analysis']
        print(f"\nDuration Analysis:")
        print(f"  Average Duration: {duration['avg_duration_hours']:.1f} hours")
        print(f"  Median Duration: {duration['median_duration_hours']:.1f} hours")
        
        print("="*60)