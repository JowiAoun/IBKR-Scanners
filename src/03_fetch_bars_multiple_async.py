"""
Fetches multiple bars asynchronously. Try with "AAPL TSLA MSFT QCOM GOOG" (CLI args)
This is much faster, because it is asynchronous.
"""

import argparse
import asyncio
import time
from ib_async import IB
from ib_async.contract import Stock
from utils.ibkr_connection import get_ibkr_connection_async

async def fetch_data(ib: IB, symbol: str):
    print(f"== Requesting data for {symbol} ==")

    start = time.perf_counter()

    contract = Stock(symbol, "SMART", "USD")
    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime="",
        durationStr="30 D",
        barSizeSetting="5 mins",
        whatToShow="TRADES",
        useRTH=True
    )

    # print bars for symbol
    print(f"=== Received {symbol} Bars ===")

    for bar in bars[-10:]:
        print(f"{bar.date} O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f} V={int(bar.volume)}")

    end = time.perf_counter()  # ⏱️ End timing
    print(f"Finished fetching {symbol} in {end - start} seconds")

# Main function that connects once and launches all requests concurrently
async def main(symbols):
    ib = await get_ibkr_connection_async()

    start = time.perf_counter()

    # Launch fetch_data tasks sequentially using the same connection
    tasks = []
    for symbol in symbols:
        task = fetch_data(ib, symbol)
        tasks.append(task)

    await asyncio.gather(*tasks)

    end = time.perf_counter()

    print(f"Finished fetching {len(symbols)} symbols in {end - start:.2f} seconds")

    ib.disconnect()

# start program
if __name__ == "__main__":
    # Parse command line arguments
    p = argparse.ArgumentParser(description="Fetch 1-min bars for multiple symbols from IBKR")
    p.add_argument("symbols", nargs="+", help="One or more ticker symbols, e.g. AAPL MSFT TSLA")
    args = p.parse_args()

    asyncio.run(main(args.symbols))