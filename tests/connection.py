import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.ibkr_connection import get_ibkr_connection_with_prompt

try:
    ib = get_ibkr_connection_with_prompt()
    print("Connection test completed successfully!")
except ConnectionError as e:
    print(f"✗ {e}")

ib.disconnect()