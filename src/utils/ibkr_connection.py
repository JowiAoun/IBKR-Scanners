import os
from typing import Optional
from ib_async import IB
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_ibkr_connection(
    host: Optional[str] = None,
    port: int = 7497,
    client_id: int = 1,
    timeout: int = 15
) -> IB:
    """
    Create and return a connected IBKR IB instance.
    
    Args:
        host: IBKR host IP. If None, uses IBKR_HOST env var (required)
        port: IBKR port (default: 7497)
        client_id: Client ID for the connection (default: 1)
        timeout: Connection timeout in seconds (default: 15)
    
    Returns:
        Connected IB instance
        
    Raises:
        ConnectionError: If connection fails
    """
    if host is None:
        host = os.getenv('IBKR_HOST')
        if not host:
            raise ValueError("IBKR_HOST environment variable is required when host is not specified")
    
    ib = IB()
    
    try:
        ib.connect(host, port, clientId=client_id, timeout=timeout)
        if not ib.isConnected():
            raise ConnectionError(f"Failed to connect to IBKR at {host}:{port}")
        return ib
    except Exception as e:
        raise ConnectionError(f"IBKR connection failed: {str(e)}")


def get_ibkr_connection_with_prompt(
    host: Optional[str] = None,
    port: int = 7497,
    client_id: int = 1,
    timeout: int = 15
) -> IB:
    """
    Create and return a connected IBKR IB instance with user prompts for TWS acceptance.
    
    Args:
        host: IBKR host IP. If None, uses IBKR_HOST env var (required)
        port: IBKR port (default: 7497)
        client_id: Client ID for the connection (default: 1)
        timeout: Connection timeout in seconds (default: 15)
    
    Returns:
        Connected IB instance
        
    Raises:
        ConnectionError: If connection fails
    """
    if host is None:
        host = os.getenv('IBKR_HOST')
        if not host:
            raise ValueError("IBKR_HOST environment variable is required when host is not specified")
    
    print("Attempting to connect to TWS...")
    print("Please accept the API connection prompt in TWS when it appears...")
    
    ib = IB()
    
    try:
        ib.connect(host, port, clientId=client_id, timeout=timeout)
        
        if ib.isConnected():
            print("✓ Successfully connected to IBKR API!")
            print(f"Server version: {ib.client.serverVersion()}")
            return ib
        else:
            raise ConnectionError(f"Failed to connect to IBKR at {host}:{port}")
            
    except Exception as e:
        raise ConnectionError(f"IBKR connection failed: {str(e)}")


async def get_ibkr_connection_async(
    host: Optional[str] = None,
    port: int = 7497,
    client_id: int = 1,
    timeout: int = 15
) -> IB:
    """
    Create and return a connected IBKR IB instance asynchronously.
    
    Args:
        host: IBKR host IP. If None, uses IBKR_HOST env var (required)
        port: IBKR port (default: 7497)
        client_id: Client ID for the connection (default: 1)
        timeout: Connection timeout in seconds (default: 15)
    
    Returns:
        Connected IB instance
        
    Raises:
        ConnectionError: If connection fails
    """
    if host is None:
        host = os.getenv('IBKR_HOST')
        if not host:
            raise ValueError("IBKR_HOST environment variable is required when host is not specified")
    
    ib = IB()
    
    try:
        await ib.connectAsync(host, port, clientId=client_id, timeout=timeout)
        if not ib.isConnected():
            raise ConnectionError(f"Failed to connect to IBKR at {host}:{port}")
        return ib
    except Exception as e:
        raise ConnectionError(f"IBKR connection failed: {str(e)}")


async def get_ibkr_connection_with_prompt_async(
    host: Optional[str] = None,
    port: int = 7497,
    client_id: int = 1,
    timeout: int = 15
) -> IB:
    """
    Create and return a connected IBKR IB instance asynchronously with user prompts for TWS acceptance.
    
    Args:
        host: IBKR host IP. If None, uses IBKR_HOST env var (required)
        port: IBKR port (default: 7497)
        client_id: Client ID for the connection (default: 1)
        timeout: Connection timeout in seconds (default: 15)
    
    Returns:
        Connected IB instance
        
    Raises:
        ConnectionError: If connection fails
    """
    if host is None:
        host = os.getenv('IBKR_HOST')
        if not host:
            raise ValueError("IBKR_HOST environment variable is required when host is not specified")
    
    print("Attempting to connect to TWS...")
    print("Please accept the API connection prompt in TWS when it appears...")
    
    ib = IB()
    
    try:
        await ib.connectAsync(host, port, clientId=client_id, timeout=timeout)
        
        if ib.isConnected():
            print("✓ Successfully connected to IBKR API!")
            print(f"Server version: {ib.client.serverVersion()}")
            return ib
        else:
            raise ConnectionError(f"Failed to connect to IBKR at {host}:{port}")
            
    except Exception as e:
        raise ConnectionError(f"IBKR connection failed: {str(e)}")