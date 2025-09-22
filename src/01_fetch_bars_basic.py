"""
Fetches bars for a single symbol synchronously.
The speed is not very noticeable because it is not much data to fetch.
See 02_fetch_bars_multiple_sync.py for a noticeable difference.
"""

from ib_async.contract import Stock
from utils.ibkr_connection import get_ibkr_connection

ib = get_ibkr_connection()

contract = Stock("AAPL", "SMART", "USD")

bars = ib.reqHistoricalData(
    contract,
    endDateTime="",
    durationStr="1 D",
    barSizeSetting="1 min",
    whatToShow="TRADES",
    useRTH=True
)

# Print the first 15 bars (first 15 minutes as they are 1 minute bars)
for bar in bars[15:]:
    print(
        f"{bar.date}  "
        f"O={bar.open}  "
        f"H={bar.high}  "
        f"L={bar.low}  "
        f"C={bar.close}  "
        f"V={int(bar.volume)}"
    )

ib.disconnect()