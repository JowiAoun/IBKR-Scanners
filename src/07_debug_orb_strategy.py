"""
Debug Opening Range Breakout Strategy

This script runs the ORB strategy with debug mode enabled to see why trades aren't being generated.
Useful for troubleshooting strategy logic.

Usage:
    python 07_debug_orb_strategy.py [SYMBOL]

Example:
    python 07_debug_orb_strategy.py AAPL
"""

import asyncio
import argparse
import sys
import os
import pandas as pd

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtester.backtester import Backtester
from strategies.opening_range_breakout import OpeningRangeBreakout


async def debug_orb_strategy(symbol="AAPL", duration="5 D", bar_size="1 min"):
    """
    Run ORB strategy with debug mode to analyze why trades aren't generated.
    
    Args:
        symbol: Single symbol to analyze
        duration: Data duration
        bar_size: Bar size
    """
    print(f"🔍 DEBUGGING ORB STRATEGY FOR {symbol}")
    print("=" * 60)
    
    # Initialize backtester
    backtester = Backtester(
        initial_capital=100000,
        commission_per_share=0.005,
        slippage_bps=2.0
    )
    
    # Create ORB strategy with debug mode enabled
    orb_strategy = OpeningRangeBreakout(
        opening_range_minutes=30,
        min_range_percent=0.3,      # Lower threshold
        max_range_percent=8.0,      # Higher threshold  
        position_size_percent=2.0,
        stop_loss_percent=2.0,
        profit_target_percent=4.0,
        max_positions=1             # Only one position for debugging
    )
    
    # Enable debug mode
    orb_strategy.set_state('debug_mode', True)
    
    backtester.add_strategy(orb_strategy)
    
    # Load data for single symbol
    print(f"Loading {duration} of data for {symbol}...")
    try:
        await backtester.load_data([symbol], duration, bar_size, delay_between_requests=1.0)
        print(f"✓ Loaded data for {symbol}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Analyze the loaded data
    data = backtester.historical_data[symbol]
    print(f"\n📊 DATA ANALYSIS:")
    print(f"Total bars: {len(data)}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
    
    # Show opening range statistics
    print(f"\n📈 OPENING RANGE ANALYSIS:")
    or_summary = orb_strategy.get_opening_ranges_summary()
    if or_summary and symbol in or_summary:
        stats = or_summary[symbol]
        print(f"Valid trading days: {stats['total_days']}")
        print(f"Average range: {stats['avg_range_percent']:.2f}%")
        print(f"Range spread: {stats['min_range_percent']:.2f}% - {stats['max_range_percent']:.2f}%")
    else:
        print("❌ No valid opening ranges found!")
        
    # Show first few opening ranges in detail
    if symbol in orb_strategy.state['opening_ranges']:
        ranges = orb_strategy.state['opening_ranges'][symbol]
        print(f"\n🎯 DETAILED OPENING RANGES (first 5):")
        for i, (date, range_data) in enumerate(list(ranges.items())[:5]):
            print(f"  {date}: ${range_data['low']:.2f} - ${range_data['high']:.2f} "
                  f"({range_data['range_percent']:.2f}%) "
                  f"from {range_data['start_time'].strftime('%H:%M')} to {range_data['end_time'].strftime('%H:%M')}")
    
    # Run backtest with limited time range for detailed analysis
    print(f"\n🚀 RUNNING DEBUG BACKTEST...")
    print("Debug output will show strategy decision-making process:")
    print("-" * 60)
    
    results = backtester.run_backtest()
    
    print("-" * 60)
    print(f"\n📋 BACKTEST SUMMARY:")
    print(f"Total trades: {len(backtester.portfolio.trades)}")
    print(f"Signals generated: {len(backtester.signals_log)}")
    print(f"Final equity: ${backtester.portfolio.total_equity:,.2f}")
    
    # Analyze debug log
    if 'debug_log' in orb_strategy.state and orb_strategy.state['debug_log']:
        debug_df = pd.DataFrame(orb_strategy.state['debug_log'])
        
        print(f"\n🔍 DEBUG ANALYSIS ({len(debug_df)} data points analyzed):")
        
        # Count reasons for no signals
        no_range_count = debug_df['no_opening_range_reason'].notna().sum() if 'no_opening_range_reason' in debug_df else 0
        before_range_count = debug_df.get('before_range_end', False).sum()
        zero_position_count = debug_df.get('position_size_zero', False).sum()
        breakout_count = (debug_df.get('bullish_breakout', False) | debug_df.get('bearish_breakout', False)).sum()
        
        print(f"  Times no opening range: {no_range_count}")
        print(f"  Times before range end: {before_range_count}")
        print(f"  Times position size zero: {zero_position_count}")
        print(f"  Times breakout detected: {breakout_count}")
        
        if breakout_count > 0:
            print(f"\n🎯 BREAKOUT OPPORTUNITIES FOUND: {breakout_count}")
            breakout_data = debug_df[
                (debug_df.get('bullish_breakout', False) | debug_df.get('bearish_breakout', False))
            ]
            for idx, row in breakout_data.head(10).iterrows():
                direction = "BULLISH" if row.get('bullish_breakout', False) else "BEARISH"
                print(f"  {row['timestamp']}: {direction} breakout at ${row['current_price']:.2f} "
                      f"(range: ${row.get('or_low', 0):.2f}-${row.get('or_high', 0):.2f})")
        else:
            print(f"\n❌ NO BREAKOUTS DETECTED")
            print("This suggests:")
            print("  - Price ranges are too narrow for breakouts")
            print("  - Breakout thresholds might be too strict")
            print("  - Market conditions are choppy/sideways")
    
    # Show signal log
    if backtester.signals_log:
        signals_df = pd.DataFrame(backtester.signals_log)
        print(f"\n📝 SIGNALS LOG:")
        print(signals_df.to_string())
    else:
        print(f"\n❌ NO SIGNALS GENERATED")
    
    return results


async def main():
    """Main debug function."""
    parser = argparse.ArgumentParser(description="Debug ORB strategy for single symbol")
    parser.add_argument("symbol", nargs="?", default="AAPL", 
                       help="Symbol to debug (default: AAPL)")
    parser.add_argument("--duration", type=str, default="5 D", 
                       help="Data duration (default: '5 D')")
    
    args = parser.parse_args()
    
    try:
        await debug_orb_strategy(args.symbol, args.duration)
        
    except KeyboardInterrupt:
        print("\n⚠️  Debug interrupted by user")
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("ORB Strategy Debugger")
    print("=" * 30)
    
    # Check required packages
    try:
        import pandas
        import numpy
        print("✓ Required packages available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        sys.exit(1)
    
    # Run debug
    asyncio.run(main())