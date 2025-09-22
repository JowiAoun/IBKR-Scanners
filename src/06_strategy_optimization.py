"""
Strategy Parameter Optimization

This script demonstrates how to optimize strategy parameters using
a grid search approach to find the best performing parameter combinations.

Usage:
    python 06_strategy_optimization.py [SYMBOLS...]

Example:
    python 06_strategy_optimization.py AAPL TSLA
"""

import asyncio
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import product
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtester.backtester import Backtester
from backtester.performance_analyzer import PerformanceAnalyzer
from strategies.opening_range_breakout import OpeningRangeBreakout


class StrategyOptimizer:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.optimization_results = []
        
    async def load_data_once(self, symbols, duration="30 D", bar_size="1 min"):
        """Load data once and reuse for all optimizations."""
        print("Loading historical data for optimization...")
        
        # Create a temporary backtester just to load data
        temp_backtester = Backtester(initial_capital=self.initial_capital)
        await temp_backtester.load_data(symbols, duration, bar_size)
        
        self.historical_data = temp_backtester.historical_data
        self.symbols = symbols
        
        print(f"✓ Loaded data for {len(self.historical_data)} symbols")
        return len(self.historical_data) > 0
    
    async def test_parameter_combination(self, param_combo):
        """Test a single parameter combination."""
        # Unpack parameters
        (opening_range_minutes, min_range_percent, max_range_percent, 
         position_size_percent, stop_loss_percent, profit_target_percent, 
         max_positions) = param_combo
        
        try:
            # Create backtester
            backtester = Backtester(
                initial_capital=self.initial_capital,
                commission_per_share=0.005,
                slippage_bps=2.0
            )
            
            # Create strategy with current parameters
            strategy = OpeningRangeBreakout(
                opening_range_minutes=opening_range_minutes,
                min_range_percent=min_range_percent,
                max_range_percent=max_range_percent,
                position_size_percent=position_size_percent,
                stop_loss_percent=stop_loss_percent,
                profit_target_percent=profit_target_percent,
                max_positions=max_positions
            )
            
            backtester.add_strategy(strategy)
            
            # Set the pre-loaded data
            backtester.set_data(self.historical_data)
            
            # Run backtest
            results = backtester.run_backtest()
            
            # Extract key metrics
            summary = results['summary']
            analyzer = PerformanceAnalyzer(results)
            
            # Calculate additional metrics
            sharpe_ratio = analyzer.calculate_sharpe_ratio()
            max_drawdown_info = analyzer.calculate_max_drawdown()
            win_rate_metrics = analyzer.calculate_win_rate_metrics()
            
            result = {
                # Parameters
                'opening_range_minutes': opening_range_minutes,
                'min_range_percent': min_range_percent,
                'max_range_percent': max_range_percent,
                'position_size_percent': position_size_percent,
                'stop_loss_percent': stop_loss_percent,
                'profit_target_percent': profit_target_percent,
                'max_positions': max_positions,
                
                # Performance metrics
                'total_return': summary.get('total_return', 0),
                'final_equity': summary.get('final_equity', self.initial_capital),
                'total_trades': summary.get('total_trades', 0),
                'win_rate': win_rate_metrics.get('win_rate', 0),
                'profit_factor': win_rate_metrics.get('profit_factor', 0),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown_info.get('max_drawdown', 0),
                'expectancy': win_rate_metrics.get('expectancy', 0),
                
                # Additional metrics
                'avg_win': win_rate_metrics.get('avg_win', 0),
                'avg_loss': win_rate_metrics.get('avg_loss', 0),
                'largest_win': win_rate_metrics.get('largest_win', 0),
                'largest_loss': win_rate_metrics.get('largest_loss', 0)
            }
            
            return result
            
        except Exception as e:
            print(f"Error testing parameters {param_combo}: {e}")
            return None
    
    async def optimize_parameters(self, parameter_ranges, max_combinations=None):
        """
        Optimize strategy parameters using grid search.
        
        Args:
            parameter_ranges: Dictionary of parameter ranges
            max_combinations: Maximum number of combinations to test (None = all)
        """
        print("Starting parameter optimization...")
        print(f"Parameter ranges: {parameter_ranges}")
        
        # Generate all parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        all_combinations = list(product(*param_values))
        
        total_combinations = len(all_combinations)
        print(f"Total parameter combinations: {total_combinations}")
        
        # Limit combinations if specified
        if max_combinations and total_combinations > max_combinations:
            print(f"Limiting to {max_combinations} randomly selected combinations")
            import random
            all_combinations = random.sample(all_combinations, max_combinations)
        
        # Test each combination
        results = []
        for i, combo in enumerate(all_combinations):
            print(f"Testing combination {i+1}/{len(all_combinations)}: {dict(zip(param_names, combo))}")
            
            result = await self.test_parameter_combination(combo)
            if result:
                results.append(result)
            
            # Progress update
            if (i + 1) % 10 == 0 or i == len(all_combinations) - 1:
                print(f"Progress: {i+1}/{len(all_combinations)} ({(i+1)/len(all_combinations)*100:.1f}%)")
        
        self.optimization_results = results
        print(f"✓ Optimization complete. Tested {len(results)} valid combinations.")
        
        return results
    
    def analyze_optimization_results(self, sort_by='total_return', top_n=10):
        """
        Analyze optimization results and show top performers.
        
        Args:
            sort_by: Metric to sort by
            top_n: Number of top results to show
        """
        if not self.optimization_results:
            print("No optimization results available")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(self.optimization_results)
        
        # Sort by specified metric (descending for most metrics)
        ascending = sort_by in ['max_drawdown']  # Only drawdown should be ascending (less negative is better)
        df_sorted = df.sort_values(by=sort_by, ascending=ascending)
        
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION RESULTS (Top {top_n} by {sort_by})")
        print(f"{'='*80}")
        
        # Show top N results
        top_results = df_sorted.head(top_n)
        
        for i, (idx, row) in enumerate(top_results.iterrows()):
            print(f"\nRank {i+1}:")
            print(f"  Parameters:")
            print(f"    Opening Range: {row['opening_range_minutes']} min")
            print(f"    Range Filter: {row['min_range_percent']:.1f}% - {row['max_range_percent']:.1f}%")
            print(f"    Position Size: {row['position_size_percent']:.1f}%")
            print(f"    Stop Loss: {row['stop_loss_percent']:.1f}%")
            print(f"    Profit Target: {row['profit_target_percent']:.1f}%")
            print(f"    Max Positions: {row['max_positions']}")
            print(f"  Performance:")
            print(f"    Total Return: {row['total_return']:.2f}%")
            print(f"    Sharpe Ratio: {row['sharpe_ratio']:.2f}")
            print(f"    Win Rate: {row['win_rate']:.1f}%")
            print(f"    Profit Factor: {row['profit_factor']:.2f}")
            print(f"    Max Drawdown: {row['max_drawdown']:.2f}%")
            print(f"    Total Trades: {int(row['total_trades'])}")
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("OPTIMIZATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total combinations tested: {len(df)}")
        print(f"Best {sort_by}: {df_sorted.iloc[0][sort_by]:.2f}")
        print(f"Worst {sort_by}: {df_sorted.iloc[-1][sort_by]:.2f}")
        print(f"Average {sort_by}: {df[sort_by].mean():.2f}")
        print(f"Std Dev {sort_by}: {df[sort_by].std():.2f}")
        
        # Parameter analysis
        print(f"\nParameter Impact Analysis:")
        numeric_params = ['opening_range_minutes', 'min_range_percent', 'max_range_percent', 
                         'position_size_percent', 'stop_loss_percent', 'profit_target_percent', 'max_positions']
        
        for param in numeric_params:
            correlation = df[param].corr(df[sort_by])
            print(f"  {param} correlation with {sort_by}: {correlation:.3f}")
        
        # Save results
        df_sorted.to_csv('optimization_results.csv', index=False)
        print(f"\n✓ Full results saved to: optimization_results.csv")
        
        return df_sorted
    
    def get_best_parameters(self, metric='total_return'):
        """Get the best parameter set based on specified metric."""
        if not self.optimization_results:
            return None
        
        df = pd.DataFrame(self.optimization_results)
        ascending = metric in ['max_drawdown']
        best_row = df.sort_values(by=metric, ascending=ascending).iloc[0]
        
        return {
            'opening_range_minutes': int(best_row['opening_range_minutes']),
            'min_range_percent': best_row['min_range_percent'],
            'max_range_percent': best_row['max_range_percent'],
            'position_size_percent': best_row['position_size_percent'],
            'stop_loss_percent': best_row['stop_loss_percent'],
            'profit_target_percent': best_row['profit_target_percent'],
            'max_positions': int(best_row['max_positions'])
        }


async def main():
    """Main optimization function."""
    parser = argparse.ArgumentParser(description="Optimize Opening Range Breakout strategy parameters")
    parser.add_argument("symbols", nargs="*", default=["AAPL", "TSLA"], 
                       help="Ticker symbols to optimize on (default: AAPL TSLA)")
    parser.add_argument("--capital", type=float, default=100000, 
                       help="Initial capital (default: 100000)")
    parser.add_argument("--duration", type=str, default="30 D", 
                       help="Data duration (default: '30 D')")
    parser.add_argument("--max-combinations", type=int, default=50, 
                       help="Maximum combinations to test (default: 50)")
    
    args = parser.parse_args()
    
    try:
        # Initialize optimizer
        optimizer = StrategyOptimizer(initial_capital=args.capital)
        
        # Load data
        success = await optimizer.load_data_once(args.symbols, args.duration)
        if not success:
            print("❌ Failed to load data")
            return
        
        # Define parameter ranges for optimization
        parameter_ranges = {
            'opening_range_minutes': [15, 30, 45, 60],  # Opening range duration
            'min_range_percent': [0.3, 0.5, 0.7],      # Minimum range filter
            'max_range_percent': [3.0, 5.0, 7.0],      # Maximum range filter
            'position_size_percent': [1.0, 2.0, 3.0],  # Position sizing
            'stop_loss_percent': [1.5, 2.0, 2.5],      # Stop loss
            'profit_target_percent': [3.0, 4.0, 5.0],  # Profit target
            'max_positions': [2, 3, 4]                 # Max concurrent positions
        }
        
        # Run optimization
        results = await optimizer.optimize_parameters(
            parameter_ranges, 
            max_combinations=args.max_combinations
        )
        
        if results:
            # Analyze results
            df_results = optimizer.analyze_optimization_results()
            
            # Show best parameters for different metrics
            metrics = ['total_return', 'sharpe_ratio', 'profit_factor']
            print(f"\n{'='*80}")
            print("BEST PARAMETERS BY METRIC")
            print(f"{'='*80}")
            
            for metric in metrics:
                best_params = optimizer.get_best_parameters(metric)
                print(f"\nBest for {metric}:")
                for param, value in best_params.items():
                    print(f"  {param}: {value}")
        
        else:
            print("❌ No valid optimization results")
    
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Strategy Parameter Optimizer")
    print("=" * 40)
    
    # Check if required packages are available
    try:
        import pandas
        import numpy
        print("✓ All required packages available")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install missing packages:")
        print("pip install pandas numpy matplotlib seaborn")
        sys.exit(1)
    
    # Run optimization
    asyncio.run(main())