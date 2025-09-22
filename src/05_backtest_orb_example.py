"""
Opening Range Breakout Backtest Example

This script demonstrates how to use the backtesting framework to test
the Opening Range Breakout strategy on multiple symbols.

Usage:
    python 05_backtest_orb_example.py [SYMBOLS...]

Example:
    python 05_backtest_orb_example.py AAPL TSLA MSFT
"""

import asyncio
import argparse
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtester.backtester import Backtester
from backtester.performance_analyzer import PerformanceAnalyzer
from backtester.visualizer import BacktestVisualizer
from strategies.opening_range_breakout import OpeningRangeBreakout


async def run_orb_backtest(symbols, 
                          initial_capital=100000,
                          duration="30 D",
                          bar_size="1 min"):
    """
    Run Opening Range Breakout backtest.
    
    Args:
        symbols: List of symbols to backtest
        initial_capital: Starting capital
        duration: Data duration to fetch
        bar_size: Bar size for data
    """
    print(f"Starting ORB backtest for symbols: {symbols}")
    print(f"Initial capital: ${initial_capital:,}")
    print(f"Data period: {duration}")
    print("="*60)
    
    # Initialize backtester
    backtester = Backtester(
        initial_capital=initial_capital,
        commission_per_share=0.005,  # $0.005 per share
        slippage_bps=2.0  # 2 basis points slippage
    )
    
    # Add Opening Range Breakout strategy
    orb_strategy = OpeningRangeBreakout(
        opening_range_minutes=30,      # 30-minute opening range
        min_range_percent=0.5,         # Minimum 0.5% range
        max_range_percent=5.0,         # Maximum 5% range
        position_size_percent=2.0,     # 2% of portfolio per position
        stop_loss_percent=2.0,         # 2% stop loss
        profit_target_percent=4.0,     # 4% profit target
        max_positions=3                # Maximum 3 concurrent positions
    )
    
    backtester.add_strategy(orb_strategy)
    
    # Load historical data with delay to avoid rate limits
    print("Loading historical data...")
    try:
        await backtester.load_data(symbols, duration, bar_size, delay_between_requests=2.0)
        print(f"✓ Loaded data for {len(backtester.historical_data)} symbols")
    except ValueError as e:
        print(f"❌ {e}")
        print("This could be due to:")
        print("  - IBKR connection issues")
        print("  - Invalid symbols")
        print("  - Rate limiting (try again in a few minutes)")
        return None
    except Exception as e:
        print(f"❌ Unexpected error loading data: {e}")
        return None
    
    # Display opening range statistics
    print("\nOpening Range Statistics:")
    or_summary = orb_strategy.get_opening_ranges_summary()
    if or_summary:
        for symbol, stats in or_summary.items():
            print(f"  {symbol}: {stats['total_days']} days, "
                  f"avg range: {stats['avg_range_percent']:.2f}%")
    else:
        print("  No valid opening ranges found (check range filters and data quality)")
    
    # Run backtest
    print("\nRunning backtest...")
    try:
        results = backtester.run_backtest()
        return results
    except Exception as e:
        print(f"❌ Error during backtesting: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_results(results):
    """Analyze and display backtest results."""
    if not results:
        return
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    # Performance analysis
    analyzer = PerformanceAnalyzer(results)
    analyzer.print_performance_summary()
    
    # Generate detailed report
    report = analyzer.generate_performance_report()
    
    # Save trade details
    trades_df = results.get('trades')
    if not trades_df.empty:
        print(f"\nTrade Details (Last 10 trades):")
        print(trades_df[['symbol', 'entry_time', 'exit_time', 'pnl', 'pnl_percent']].tail(10).to_string())
        
        # Save trades to CSV
        trades_df.to_csv('orb_backtest_trades.csv', index=False)
        print(f"\n✓ All trades saved to: orb_backtest_trades.csv")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = BacktestVisualizer(results, save_dir='backtest_charts')
    figures = visualizer.create_full_report(show_plots=False)
    
    print(f"✓ Generated {len(figures)} charts saved to: backtest_charts/")
    
    return report


async def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Opening Range Breakout backtest")
    parser.add_argument("symbols", nargs="*", default=["AAPL", "TSLA", "MSFT"], 
                       help="Ticker symbols to backtest (default: AAPL TSLA MSFT)")
    parser.add_argument("--capital", type=float, default=100000, 
                       help="Initial capital (default: 100000)")
    parser.add_argument("--duration", type=str, default="30 D", 
                       help="Data duration (default: '30 D')")
    parser.add_argument("--bar-size", type=str, default="1 min", 
                       help="Bar size (default: '1 min')")
    
    args = parser.parse_args()
    
    try:
        # Run backtest
        results = await run_orb_backtest(
            symbols=args.symbols,
            initial_capital=args.capital,
            duration=args.duration,
            bar_size=args.bar_size
        )
        
        # Analyze results
        if results:
            analyze_results(results)
            
            # Quick summary
            summary = results['summary']
            print(f"\n🎯 QUICK SUMMARY:")
            print(f"   Total Return: {summary.get('total_return', 0):.2f}%")
            print(f"   Total Trades: {summary.get('total_trades', 0)}")
            print(f"   Win Rate: {summary.get('win_rate', 0):.1f}%")
            print(f"   Final Equity: ${summary.get('final_equity', 0):,.2f}")
            
        else:
            print("❌ Backtest failed to complete")
    
    except KeyboardInterrupt:
        print("\n⚠️  Backtest interrupted by user")
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Opening Range Breakout Backtester")
    print("=" * 40)
    
    # Check if required packages are available
    try:
        import pandas
        import numpy
        import matplotlib
        print("✓ All required packages available")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install missing packages:")
        print("pip install pandas numpy matplotlib seaborn")
        sys.exit(1)
    
    # Run main
    asyncio.run(main())