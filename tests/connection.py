from ib_insync import *

ib = IB()

print("Attempting to connect to TWS...")
print("Please accept the API connection prompt in TWS when it appears...")

# Connect with a longer timeout to allow time for user to accept the prompt
ib.connect('172.25.160.1', 7497, clientId=1, timeout=15)

if ib.isConnected():
    print("✓ Successfully connected to IBKR API!")
    print(f"Server version: {ib.client.serverVersion()}")
    print("Connection test completed successfully!")
else:
    print("✗ Failed to establish connection")

ib.disconnect()