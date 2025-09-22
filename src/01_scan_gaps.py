from ib_async import IB
from ib_async.contract import Stock

ib = IB()
ib.connect("172.25.160.1", 7497, clientId=1, timeout=15)

contract = Stock("AAPL", "SMART", "USD")

bars = ib.reqHistoricalData(
    contract,
    endDateTime="",
    durationStr="1 D",
    barSizeSetting="1 min",
    whatToShow="TRADES",
    useRTH=True
)

# Print all of the bars, first 15 bars, and the last 15 bars
for bar in bars[-15:]:
    print(
        f"{bar.date}  "
        f"O={bar.open}  "
        f"H={bar.high}  "
        f"L={bar.low}  "
        f"C={bar.close}  "
        f"V={int(bar.volume)}"
    )

ib.disconnect()