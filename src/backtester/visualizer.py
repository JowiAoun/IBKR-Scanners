import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os


class BacktestVisualizer:
    def __init__(self, results: Dict, save_dir: Optional[str] = None):
        """
        Initialize visualizer with backtest results.
        
        Args:
            results: Dictionary containing backtest results
            save_dir: Directory to save plots (optional)
        """
        self.results = results
        self.save_dir = save_dir
        self.equity_curve = results.get('equity_curve', pd.DataFrame())
        self.trades = results.get('trades', pd.DataFrame())
        self.summary = results.get('summary', {})
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def plot_equity_curve(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot equity curve with drawdown.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Figure object
        """
        if self.equity_curve.empty:
            print("No equity curve data available")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity curve
        equity_df = self.equity_curve.copy()
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        ax1.plot(equity_df['timestamp'], equity_df['equity'], 
                linewidth=1.5, label='Portfolio Equity', color='blue')
        ax1.axhline(y=self.summary.get('initial_capital', 100000), 
                   color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        
        ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Equity ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Drawdown
        equity_df['peak'] = equity_df['equity'].expanding().max()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        
        ax2.fill_between(equity_df['timestamp'], equity_df['drawdown'], 0, 
                        alpha=0.7, color='red', label='Drawdown')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'equity_curve.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_trade_analysis(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot comprehensive trade analysis.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Figure object
        """
        if self.trades.empty:
            print("No trade data available")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Trade Analysis Dashboard', fontsize=16, fontweight='bold')
        
        trades_df = self.trades.copy()
        
        # 1. P&L Distribution
        ax1 = axes[0, 0]
        trades_df['pnl'].hist(bins=30, alpha=0.7, ax=ax1, color='skyblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('P&L Distribution')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        
        # 2. Cumulative P&L
        ax2 = axes[0, 1]
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        ax2.plot(range(len(trades_df)), trades_df['cumulative_pnl'], 
                linewidth=2, color='green')
        ax2.set_title('Cumulative P&L')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative P&L ($)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Win/Loss by Symbol
        ax3 = axes[0, 2]
        symbol_stats = trades_df.groupby('symbol').agg({
            'pnl': ['count', lambda x: (x > 0).sum()]
        }).round(2)
        symbol_stats.columns = ['Total', 'Wins']
        symbol_stats['Win_Rate'] = (symbol_stats['Wins'] / symbol_stats['Total'] * 100).round(1)
        
        if len(symbol_stats) > 0:
            symbol_stats['Win_Rate'].plot(kind='bar', ax=ax3, color='lightcoral')
            ax3.set_title('Win Rate by Symbol')
            ax3.set_xlabel('Symbol')
            ax3.set_ylabel('Win Rate (%)')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Trade Duration Distribution
        ax4 = axes[1, 0]
        if 'duration' in trades_df.columns:
            duration_hours = trades_df['duration'].dt.total_seconds() / 3600
            duration_hours.hist(bins=20, alpha=0.7, ax=ax4, color='orange', edgecolor='black')
            ax4.set_title('Trade Duration Distribution')
            ax4.set_xlabel('Duration (Hours)')
            ax4.set_ylabel('Frequency')
        
        # 5. Monthly P&L
        ax5 = axes[1, 1]
        if 'exit_time' in trades_df.columns:
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            trades_df['month'] = trades_df['exit_time'].dt.to_period('M')
            monthly_pnl = trades_df.groupby('month')['pnl'].sum()
            
            colors = ['green' if x > 0 else 'red' for x in monthly_pnl.values]
            monthly_pnl.plot(kind='bar', ax=ax5, color=colors, alpha=0.7)
            ax5.set_title('Monthly P&L')
            ax5.set_xlabel('Month')
            ax5.set_ylabel('P&L ($)')
            ax5.tick_params(axis='x', rotation=45)
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 6. P&L vs Trade Size
        ax6 = axes[1, 2]
        if 'quantity' in trades_df.columns:
            ax6.scatter(trades_df['quantity'], trades_df['pnl'], alpha=0.6, s=30)
            ax6.set_title('P&L vs Position Size')
            ax6.set_xlabel('Position Size (Shares)')
            ax6.set_ylabel('P&L ($)')
            ax6.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'trade_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_performance_metrics(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot key performance metrics as a dashboard.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Performance Metrics Dashboard', fontsize=16, fontweight='bold')
        
        # Extract metrics from summary
        total_return = self.summary.get('total_return', 0)
        win_rate = self.summary.get('win_rate', 0)
        max_drawdown = self.summary.get('max_drawdown', 0)
        profit_factor = self.summary.get('profit_factor', 0)
        
        # 1. Return vs Benchmark (assuming benchmark = 0)
        ax1 = axes[0, 0]
        categories = ['Strategy', 'Benchmark']
        returns = [total_return, 0]
        colors = ['green' if total_return > 0 else 'red', 'gray']
        ax1.bar(categories, returns, color=colors, alpha=0.7)
        ax1.set_title('Total Return Comparison')
        ax1.set_ylabel('Return (%)')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 2. Win Rate Pie Chart
        ax2 = axes[0, 1]
        if win_rate > 0:
            sizes = [win_rate, 100 - win_rate]
            labels = ['Wins', 'Losses']
            colors = ['lightgreen', 'lightcoral']
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Win Rate: {win_rate:.1f}%')
        
        # 3. Risk Metrics
        ax3 = axes[1, 0]
        risk_metrics = ['Max DD', 'Profit Factor']
        risk_values = [abs(max_drawdown), min(profit_factor, 10)]  # Cap profit factor for display
        colors = ['red', 'green']
        bars = ax3.bar(risk_metrics, risk_values, color=colors, alpha=0.7)
        ax3.set_title('Risk Metrics')
        ax3.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, [max_drawdown, profit_factor]):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 4. Monthly Returns Heatmap (if enough data)
        ax4 = axes[1, 1]
        if not self.trades.empty and 'exit_time' in self.trades.columns:
            trades_df = self.trades.copy()
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            trades_df['year'] = trades_df['exit_time'].dt.year
            trades_df['month'] = trades_df['exit_time'].dt.month
            
            monthly_returns = trades_df.groupby(['year', 'month'])['pnl'].sum().reset_index()
            
            if not monthly_returns.empty:
                pivot_table = monthly_returns.pivot(index='year', columns='month', values='pnl')
                sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='RdYlGn', 
                           center=0, ax=ax4, cbar_kws={'label': 'P&L ($)'})
                ax4.set_title('Monthly P&L Heatmap')
                ax4.set_xlabel('Month')
                ax4.set_ylabel('Year')
        
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'performance_metrics.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_daily_returns(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot daily returns distribution and time series.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Figure object
        """
        if self.equity_curve.empty:
            print("No equity curve data available")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Daily Returns Analysis', fontsize=16, fontweight='bold')
        
        equity_df = self.equity_curve.copy()
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df['daily_return'] = equity_df['equity'].pct_change() * 100
        daily_returns = equity_df['daily_return'].dropna()
        
        # 1. Returns distribution
        daily_returns.hist(bins=50, alpha=0.7, ax=ax1, color='skyblue', edgecolor='black')
        ax1.axvline(x=daily_returns.mean(), color='red', linestyle='--', 
                   label=f'Mean: {daily_returns.mean():.3f}%')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Daily Returns Distribution')
        ax1.set_xlabel('Daily Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # 2. Returns time series
        ax2.plot(equity_df['timestamp'][1:], daily_returns, linewidth=0.8, alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('Daily Returns Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Daily Return (%)')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'daily_returns.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_full_report(self, show_plots: bool = True) -> List[plt.Figure]:
        """
        Create a comprehensive visual report.
        
        Args:
            show_plots: Whether to display plots
            
        Returns:
            List of figure objects
        """
        figures = []
        
        print("Generating backtest visualization report...")
        
        # Generate all plots
        fig1 = self.plot_equity_curve()
        if fig1:
            figures.append(fig1)
        
        fig2 = self.plot_trade_analysis()
        if fig2:
            figures.append(fig2)
        
        fig3 = self.plot_performance_metrics()
        if fig3:
            figures.append(fig3)
        
        fig4 = self.plot_daily_returns()
        if fig4:
            figures.append(fig4)
        
        if show_plots:
            plt.show()
        
        print(f"Generated {len(figures)} visualization charts")
        if self.save_dir:
            print(f"Charts saved to: {self.save_dir}")
        
        return figures
    
    def save_all_plots(self, directory: str):
        """Save all plots to specified directory."""
        self.save_dir = directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        self.create_full_report(show_plots=False)
        print(f"All plots saved to {directory}")
    
    @staticmethod
    def close_all():
        """Close all matplotlib figures."""
        plt.close('all')