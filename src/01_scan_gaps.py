from ib_async import IB, ScannerSubscription
from ib_async.contract import Contract

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

sub = ScannerSubscription()
sub.instrument  = 'STK'
sub.locationCode = 'STK.US.MAJOR'
sub.scanCode    = 'HIGH_OPEN_GAP'

scan_data = ib.reqScannerData(sub)

for data in scan_data[:10]:
    print(data.rank)
    print(data.contractDetails.contract.symbol)

ib.disconnect()