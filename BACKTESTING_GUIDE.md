# IBKR-Scanners Backtesting Guide

This guide covers the comprehensive backtesting framework for trading strategies, specifically the Opening Range Breakout (ORB) strategy.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Advanced Configuration](#advanced-configuration)
- [Strategy Parameters](#strategy-parameters)
- [Configuration Files](#configuration-files)
- [Presets](#presets)
- [Output Options](#output-options)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages
pip install -r requirements.txt

# Ensure IBKR connection is configured
# Set IBKR_HOST environment variable or use .env file
```

### Run Your First Backtest
```bash
# Simple backtest with default parameters
./run ./src/05_backtest_orb_example.py AAPL TSLA MSFT
```

## 📊 Basic Usage

### Command Structure
```bash
./run ./src/05_backtest_orb_example.py [SYMBOLS...] [OPTIONS]
```

### Basic Options
```bash
# Specify symbols
./run ./src/05_backtest_orb_example.py AAPL TSLA GOOGL

# Set initial capital
./run ./src/05_backtest_orb_example.py AAPL --capital 50000

# Change data duration
./run ./src/05_backtest_orb_example.py AAPL --duration "60 D"

# Use different bar size
./run ./src/05_backtest_orb_example.py AAPL --bar-size "5 mins"
```

## ⚙️ Advanced Configuration

### Date Range Controls
```bash
# Specific date range
./run ./src/05_backtest_orb_example.py AAPL \
  --start-date 2023-01-01 \
  --end-date 2023-12-31

# Different date formats supported
./run ./src/05_backtest_orb_example.py AAPL --start-date "01/01/2023"
```

### Symbol Management
```bash
# Load symbols from file
./run ./src/05_backtest_orb_example.py --symbols-file example_symbols.txt

# Exclude specific symbols
./run ./src/05_backtest_orb_example.py AAPL TSLA MSFT --exclude-symbols TSLA

# Combine multiple sources
./run ./src/05_backtest_orb_example.py AAPL --symbols-file stocks.txt --exclude-symbols RISKY_STOCK
```

## 🎯 Strategy Parameters

### Opening Range Breakout Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--opening-range-minutes` | 30 | Duration of opening range (10-90 minutes) |
| `--min-range-percent` | 0.5 | Minimum range % to consider valid |
| `--max-range-percent` | 5.0 | Maximum range % to consider valid |
| `--position-size-percent` | 2.0 | Position size as % of portfolio |
| `--stop-loss-percent` | 2.0 | Stop loss percentage |
| `--profit-target-percent` | 4.0 | Profit target percentage |
| `--max-positions` | 3 | Maximum concurrent positions |

### Example Strategy Customization
```bash
# Aggressive short-term strategy
./run ./src/05_backtest_orb_example.py AAPL \
  --opening-range-minutes 15 \
  --min-range-percent 0.3 \
  --max-range-percent 8.0 \
  --position-size-percent 5.0 \
  --stop-loss-percent 3.0 \
  --profit-target-percent 6.0 \
  --max-positions 5

# Conservative long-term strategy
./run ./src/05_backtest_orb_example.py AAPL \
  --opening-range-minutes 60 \
  --min-range-percent 0.8 \
  --max-range-percent 3.0 \
  --position-size-percent 1.0 \
  --stop-loss-percent 1.5 \
  --profit-target-percent 3.0 \
  --max-positions 2
```

### Trading Environment
```bash
# Custom commission and slippage
./run ./src/05_backtest_orb_example.py AAPL \
  --commission-per-share 0.001 \
  --slippage-bps 1.0 \
  --delay-between-requests 1.0
```

## 📁 Configuration Files

### Using Configuration Files
```bash
# Load configuration from YAML file
./run ./src/05_backtest_orb_example.py --config backtest_config_example.yaml

# Override config with command line
./run ./src/05_backtest_orb_example.py --config my_config.yaml --verbose
```

### Example Configuration File
```yaml
# my_backtest_config.yaml
symbols:
  - AAPL
  - TSLA
  - GOOGL
capital: 100000
duration: "30 D"

strategy:
  opening_range_minutes: 30
  min_range_percent: 0.5
  max_range_percent: 5.0
  position_size_percent: 2.0
  stop_loss_percent: 2.0
  profit_target_percent: 4.0
  max_positions: 3

output:
  output_dir: "my_results"
  save_trades: true
  save_charts: true
  verbose: false
```

## 🎛️ Presets

### Available Presets

| Preset | Use Case | Characteristics |
|--------|----------|----------------|
| `conservative` | Stable returns | Longer ranges, smaller positions, tight stops |
| `aggressive` | High growth | Shorter ranges, larger positions, wider stops |
| `scalping` | Quick profits | Very short ranges, quick exits |
| `swing` | Multi-day holds | Long ranges, wide stops, larger targets |

### Using Presets
```bash
# Conservative approach
./run ./src/05_backtest_orb_example.py AAPL --preset conservative

# Aggressive trading
./run ./src/05_backtest_orb_example.py AAPL --preset aggressive

# Scalping strategy
./run ./src/05_backtest_orb_example.py AAPL --preset scalping

# Swing trading
./run ./src/05_backtest_orb_example.py AAPL --preset swing
```

## 📈 Output Options

### Output Controls
```bash
# Save all results
./run ./src/05_backtest_orb_example.py AAPL \
  --save-trades \
  --save-charts \
  --output-dir "my_backtest_results"

# Verbose debugging
./run ./src/05_backtest_orb_example.py AAPL --verbose

# Quiet mode (minimal output)
./run ./src/05_backtest_orb_example.py AAPL --quiet
```

### Generated Files
```
backtest_results/
├── trades.csv              # Detailed trade log
├── backtest_summary.json   # Performance metrics
├── equity_curve.png        # Portfolio performance chart
├── trade_analysis.png      # Trade analysis dashboard
├── performance_metrics.png # Key metrics visualization
└── daily_returns.png       # Returns distribution
```

## 💡 Examples

### Example 1: Quick Test
```bash
# Test strategy on AAPL with default settings
./run ./src/05_backtest_orb_example.py AAPL
```

### Example 2: Custom Date Range
```bash
# Backtest specific period
./run ./src/05_backtest_orb_example.py AAPL TSLA \
  --start-date 2023-06-01 \
  --end-date 2023-12-31 \
  --save-charts
```

### Example 3: Portfolio Simulation
```bash
# Test with multiple symbols and custom parameters
./run ./src/05_backtest_orb_example.py \
  --symbols-file example_symbols.txt \
  --capital 250000 \
  --opening-range-minutes 45 \
  --position-size-percent 1.5 \
  --max-positions 5 \
  --save-trades \
  --save-charts \
  --verbose
```

### Example 4: Strategy Optimization
```bash
# Test different presets for comparison
./run ./src/05_backtest_orb_example.py AAPL --preset conservative --output-dir "results_conservative" --quiet
./run ./src/05_backtest_orb_example.py AAPL --preset aggressive --output-dir "results_aggressive" --quiet
./run ./src/05_backtest_orb_example.py AAPL --preset scalping --output-dir "results_scalping" --quiet
```

### Example 5: Configuration-Based Testing
```bash
# Create custom config file
cat > my_strategy.yaml << EOF
symbols: [AAPL, TSLA, MSFT, GOOGL]
capital: 100000
duration: "90 D"
strategy:
  opening_range_minutes: 20
  min_range_percent: 0.4
  position_size_percent: 2.5
output:
  save_trades: true
  save_charts: true
  verbose: true
EOF

# Run with config
./run ./src/05_backtest_orb_example.py --config my_strategy.yaml
```

## 🔍 Debugging and Analysis

### Debug Mode
```bash
# Enable verbose debugging to see strategy decisions
./run ./src/07_debug_orb_strategy.py AAPL

# Debug with custom parameters
./run ./src/07_debug_orb_strategy.py AAPL --duration "5 D"
```

### Strategy Optimization
```bash
# Test multiple parameter combinations
./run ./src/06_strategy_optimization.py AAPL TSLA --max-combinations 50
```

## 🔧 Troubleshooting

### Common Issues

**No trades generated:**
```bash
# Check with debug mode
./run ./src/07_debug_orb_strategy.py AAPL

# Try more permissive parameters
./run ./src/05_backtest_orb_example.py AAPL \
  --min-range-percent 0.2 \
  --max-range-percent 10.0 \
  --verbose
```

**IBKR connection issues:**
```bash
# Check environment variables
echo $IBKR_HOST

# Test with longer delays
./run ./src/05_backtest_orb_example.py AAPL --delay-between-requests 5.0
```

**Memory issues with large datasets:**
```bash
# Use shorter duration or fewer symbols
./run ./src/05_backtest_orb_example.py AAPL --duration "7 D"
```

### Performance Tips

1. **Start Small**: Begin with single symbol and short duration
2. **Use Presets**: Start with presets before custom parameters
3. **Debug First**: Use debug mode to understand strategy behavior
4. **Save Results**: Always use `--save-trades` and `--save-charts` for analysis
5. **Batch Testing**: Use quiet mode for multiple backtests

## 📞 Getting Help

### Command Line Help
```bash
# Show all available options
./run ./src/05_backtest_orb_example.py --help

# Debug specific strategy behavior
./run ./src/07_debug_orb_strategy.py --help

# Strategy optimization options
./run ./src/06_strategy_optimization.py --help
```

### Additional Resources
- Check `example_symbols.txt` for symbol examples
- Review `backtest_config_example.yaml` for configuration templates
- Examine generated JSON reports for detailed metrics
- Use visualization charts to understand strategy performance

---

## 📝 Notes

- All timestamps are handled in US/Eastern timezone
- Commission and slippage modeling provides realistic results
- Strategy parameters can be optimized using the optimization script
- Results are saved automatically for further analysis
- Configuration files support YAML format for easy editing