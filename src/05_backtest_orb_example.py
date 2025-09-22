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
                          bar_size="1 min",
                          start_date=None,
                          end_date=None,
                          # Strategy parameters
                          opening_range_minutes=30,
                          min_range_percent=0.5,
                          max_range_percent=5.0,
                          position_size_percent=2.0,
                          stop_loss_percent=2.0,
                          profit_target_percent=4.0,
                          max_positions=3,
                          # Trading environment
                          commission_per_share=0.005,
                          slippage_bps=2.0,
                          delay_between_requests=2.0,
                          # Output options
                          verbose=False):
    """
    Run Opening Range Breakout backtest.
    
    Args:
        symbols: List of symbols to backtest
        initial_capital: Starting capital
        duration: Data duration to fetch
        bar_size: Bar size for data
        start_date: Backtest start date (datetime or None)
        end_date: Backtest end date (datetime or None)
        opening_range_minutes: Opening range duration in minutes
        min_range_percent: Minimum range % to consider valid
        max_range_percent: Maximum range % to consider valid
        position_size_percent: Position size as % of portfolio
        stop_loss_percent: Stop loss percentage
        profit_target_percent: Profit target percentage
        max_positions: Maximum concurrent positions
        commission_per_share: Commission per share
        slippage_bps: Slippage in basis points
        delay_between_requests: Delay between API requests
        verbose: Enable verbose logging
    """
    print(f"Starting ORB backtest for symbols: {symbols}")
    print(f"Initial capital: ${initial_capital:,}")
    print(f"Data period: {duration}")
    print("="*60)
    
    # Initialize backtester with configurable parameters
    backtester = Backtester(
        initial_capital=initial_capital,
        commission_per_share=commission_per_share,
        slippage_bps=slippage_bps
    )
    
    # Add Opening Range Breakout strategy with configurable parameters
    orb_strategy = OpeningRangeBreakout(
        opening_range_minutes=opening_range_minutes,
        min_range_percent=min_range_percent,
        max_range_percent=max_range_percent,
        position_size_percent=position_size_percent,
        stop_loss_percent=stop_loss_percent,
        profit_target_percent=profit_target_percent,
        max_positions=max_positions
    )
    
    # Enable verbose mode if requested
    if verbose:
        orb_strategy.set_state('debug_mode', True)
    
    backtester.add_strategy(orb_strategy)
    
    # Load historical data with delay to avoid rate limits
    print("Loading historical data...")
    try:
        await backtester.load_data(symbols, duration, bar_size, delay_between_requests=delay_between_requests)
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
    
    # Run backtest with optional date range
    print("\nRunning backtest...")
    try:
        results = backtester.run_backtest(start_date=start_date, end_date=end_date)
        return results
    except Exception as e:
        print(f"❌ Error during backtesting: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_results(results, args):
    """Analyze and display backtest results with configurable output."""
    if not results:
        return
    
    if not args.quiet:
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
    
    # Performance analysis
    analyzer = PerformanceAnalyzer(results)
    if not args.quiet:
        analyzer.print_performance_summary()
    
    # Generate detailed report
    report = analyzer.generate_performance_report()
    
    # Save trade details
    trades_df = results.get('trades')
    if not trades_df.empty:
        if not args.quiet:
            print(f"\nTrade Details (Last 10 trades):")
            print(trades_df[['symbol', 'entry_time', 'exit_time', 'pnl', 'pnl_percent']].tail(10).to_string())
        
        # Save trades to CSV if requested
        if args.save_trades:
            import os
            trades_file = os.path.join(args.output_dir, 'trades.csv')
            trades_df.to_csv(trades_file, index=False)
            print(f"✓ Trades saved to: {trades_file}")
    
    # Generate visualizations if requested
    if args.save_charts:
        if not args.quiet:
            print("\nGenerating visualizations...")
        visualizer = BacktestVisualizer(results, save_dir=args.output_dir)
        figures = visualizer.create_full_report(show_plots=False)
        
        if not args.quiet:
            print(f"✓ Generated {len(figures)} charts saved to: {args.output_dir}/")
    
    # Save summary report
    import json
    import os
    summary_file = os.path.join(args.output_dir, 'backtest_summary.json')
    with open(summary_file, 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        serializable_report = {}
        for key, value in report.items():
            if isinstance(value, dict):
                serializable_report[key] = {k: str(v) if hasattr(v, 'strftime') else v for k, v in value.items()}
            else:
                serializable_report[key] = str(value) if hasattr(value, 'strftime') else value
        
        json.dump(serializable_report, f, indent=2, default=str)
    
    if not args.quiet:
        print(f"✓ Summary report saved to: {summary_file}")
    
    return report


def parse_date(date_string):
    """Parse date string in various formats."""
    if not date_string:
        return None
    
    from datetime import datetime
    
    # Try different date formats
    formats = [
        "%Y-%m-%d",           # 2023-01-15
        "%Y-%m-%d %H:%M:%S",  # 2023-01-15 09:30:00
        "%m/%d/%Y",           # 01/15/2023
        "%d/%m/%Y",           # 15/01/2023
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse date: {date_string}")


def get_symbols(args):
    """Get symbols from various sources."""
    symbols = []
    
    # From command line arguments
    if args.symbols:
        symbols.extend(args.symbols)
    
    # From symbols file
    if args.symbols_file:
        try:
            with open(args.symbols_file, 'r') as f:
                file_symbols = [line.strip().upper() for line in f if line.strip()]
                symbols.extend(file_symbols)
                print(f"✓ Loaded {len(file_symbols)} symbols from {args.symbols_file}")
        except FileNotFoundError:
            print(f"⚠️  Symbols file not found: {args.symbols_file}")
        except Exception as e:
            print(f"⚠️  Error reading symbols file: {e}")
    
    # Remove duplicates and excluded symbols
    symbols = list(set(symbols))  # Remove duplicates
    if args.exclude_symbols:
        excluded = [s.upper() for s in args.exclude_symbols]
        symbols = [s for s in symbols if s not in excluded]
        print(f"✓ Excluded {len(excluded)} symbols")
    
    if not symbols:
        symbols = ["AAPL", "TSLA", "MSFT"]  # Default fallback
        print("⚠️  No symbols specified, using default: AAPL TSLA MSFT")
    
    return symbols


def apply_preset(args):
    """Apply predefined parameter presets."""
    presets = {
        "conservative": {
            "opening_range_minutes": 60,
            "min_range_percent": 0.8,
            "max_range_percent": 3.0,
            "position_size_percent": 1.0,
            "stop_loss_percent": 1.5,
            "profit_target_percent": 3.0,
            "max_positions": 2
        },
        "aggressive": {
            "opening_range_minutes": 15,
            "min_range_percent": 0.3,
            "max_range_percent": 8.0,
            "position_size_percent": 5.0,
            "stop_loss_percent": 3.0,
            "profit_target_percent": 6.0,
            "max_positions": 5
        },
        "scalping": {
            "opening_range_minutes": 10,
            "min_range_percent": 0.2,
            "max_range_percent": 2.0,
            "position_size_percent": 3.0,
            "stop_loss_percent": 0.5,
            "profit_target_percent": 1.0,
            "max_positions": 8
        },
        "swing": {
            "opening_range_minutes": 90,
            "min_range_percent": 1.0,
            "max_range_percent": 6.0,
            "position_size_percent": 4.0,
            "stop_loss_percent": 4.0,
            "profit_target_percent": 8.0,
            "max_positions": 3
        }
    }
    
    if args.preset in presets:
        preset_params = presets[args.preset]
        print(f"✓ Applying {args.preset} preset parameters")
        
        # Apply preset parameters
        for param, value in preset_params.items():
            setattr(args, param, value)
    
    return args


def load_config(args):
    """Load configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        print("⚠️  PyYAML not installed. Install with: pip install pyyaml")
        return args
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Loaded configuration from {args.config}")
        
        # Map config file values to args
        if 'symbols' in config:
            args.symbols = config['symbols']
        if 'capital' in config:
            args.capital = config['capital']
        if 'duration' in config:
            args.duration = config['duration']
        if 'bar_size' in config:
            args.bar_size = config['bar_size']
        if 'start_date' in config:
            args.start_date = config['start_date']
        if 'end_date' in config:
            args.end_date = config['end_date']
        
        # Strategy parameters
        if 'strategy' in config:
            strategy = config['strategy']
            for key, value in strategy.items():
                setattr(args, key, value)
        
        # Trading parameters
        if 'trading' in config:
            trading = config['trading']
            for key, value in trading.items():
                setattr(args, key, value)
        
        # Output parameters
        if 'output' in config:
            output = config['output']
            for key, value in output.items():
                setattr(args, key, value)
        
        # Symbol management
        if 'symbol_management' in config:
            sm = config['symbol_management']
            if 'symbols_file' in sm:
                args.symbols_file = sm['symbols_file']
            if 'exclude_symbols' in sm:
                args.exclude_symbols = sm['exclude_symbols']
        
        # Preset
        if 'preset' in config:
            args.preset = config['preset']
        
    except FileNotFoundError:
        print(f"⚠️  Configuration file not found: {args.config}")
    except Exception as e:
        print(f"⚠️  Error loading configuration: {e}")
    
    return args


async def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run Opening Range Breakout backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL TSLA MSFT --duration "60 D"
  %(prog)s AAPL --start-date 2023-01-01 --end-date 2023-12-31
  %(prog)s AAPL --opening-range-minutes 15 --min-range-percent 0.3
  %(prog)s --symbols-file symbols.txt --verbose --save-charts
        """
    )
    
    # Basic options
    parser.add_argument("symbols", nargs="*", default=["AAPL", "TSLA", "MSFT"], 
                       help="Ticker symbols to backtest (default: AAPL TSLA MSFT)")
    parser.add_argument("--capital", type=float, default=100000, 
                       help="Initial capital (default: 100000)")
    parser.add_argument("--duration", type=str, default="30 D", 
                       help="Data duration (default: '30 D')")
    parser.add_argument("--bar-size", type=str, default="1 min", 
                       help="Bar size (default: '1 min')")
    
    # Date range controls
    parser.add_argument("--start-date", type=str, default=None,
                       help="Backtest start date (YYYY-MM-DD or MM/DD/YYYY)")
    parser.add_argument("--end-date", type=str, default=None,
                       help="Backtest end date (YYYY-MM-DD or MM/DD/YYYY)")
    
    # Strategy parameters
    strategy_group = parser.add_argument_group('Strategy Parameters')
    strategy_group.add_argument("--opening-range-minutes", type=int, default=30,
                               help="Opening range duration in minutes (default: 30)")
    strategy_group.add_argument("--min-range-percent", type=float, default=0.5,
                               help="Minimum range %% to consider valid (default: 0.5)")
    strategy_group.add_argument("--max-range-percent", type=float, default=5.0,
                               help="Maximum range %% to consider valid (default: 5.0)")
    strategy_group.add_argument("--position-size-percent", type=float, default=2.0,
                               help="Position size as %% of portfolio (default: 2.0)")
    strategy_group.add_argument("--stop-loss-percent", type=float, default=2.0,
                               help="Stop loss percentage (default: 2.0)")
    strategy_group.add_argument("--profit-target-percent", type=float, default=4.0,
                               help="Profit target percentage (default: 4.0)")
    strategy_group.add_argument("--max-positions", type=int, default=3,
                               help="Maximum concurrent positions (default: 3)")
    
    # Trading environment
    trading_group = parser.add_argument_group('Trading Environment')
    trading_group.add_argument("--commission-per-share", type=float, default=0.005,
                              help="Commission per share (default: 0.005)")
    trading_group.add_argument("--slippage-bps", type=float, default=2.0,
                              help="Slippage in basis points (default: 2.0)")
    trading_group.add_argument("--delay-between-requests", type=float, default=2.0,
                              help="Delay between IBKR requests in seconds (default: 2.0)")
    
    # Symbol management
    symbol_group = parser.add_argument_group('Symbol Management')
    symbol_group.add_argument("--symbols-file", type=str, default=None,
                             help="File containing symbols (one per line)")
    symbol_group.add_argument("--exclude-symbols", type=str, nargs="*", default=[],
                             help="Symbols to exclude from backtest")
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument("--output-dir", type=str, default="backtest_results",
                             help="Output directory for results (default: backtest_results)")
    output_group.add_argument("--save-trades", action="store_true",
                             help="Save trade details to CSV")
    output_group.add_argument("--save-charts", action="store_true",
                             help="Generate and save visualization charts")
    output_group.add_argument("--verbose", "-v", action="store_true",
                             help="Enable verbose/debug output")
    output_group.add_argument("--quiet", "-q", action="store_true",
                             help="Minimal output")
    
    # Presets and configuration
    parser.add_argument("--preset", type=str, choices=["conservative", "aggressive", "scalping", "swing"],
                       help="Use predefined parameter preset")
    parser.add_argument("--config", type=str, default=None,
                       help="Load configuration from YAML file")
    
    args = parser.parse_args()
    
    # Load configuration file if specified
    if args.config:
        args = load_config(args)
    
    try:
        # Handle presets
        if args.preset:
            args = apply_preset(args)
        
        # Handle symbol sources
        symbols = get_symbols(args)
        
        # Parse dates
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        
        # Create output directory
        import os
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Run backtest with all parameters
        results = await run_orb_backtest(
            symbols=symbols,
            initial_capital=args.capital,
            duration=args.duration,
            bar_size=args.bar_size,
            start_date=start_date,
            end_date=end_date,
            # Strategy parameters
            opening_range_minutes=args.opening_range_minutes,
            min_range_percent=args.min_range_percent,
            max_range_percent=args.max_range_percent,
            position_size_percent=args.position_size_percent,
            stop_loss_percent=args.stop_loss_percent,
            profit_target_percent=args.profit_target_percent,
            max_positions=args.max_positions,
            # Trading environment
            commission_per_share=args.commission_per_share,
            slippage_bps=args.slippage_bps,
            delay_between_requests=args.delay_between_requests,
            # Output options
            verbose=args.verbose
        )
        
        # Analyze results with enhanced output options
        if results:
            analyze_results(results, args)
            
            # Quick summary
            if not args.quiet:
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