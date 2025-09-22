from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd


@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0


@dataclass
class Trade:
    symbol: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    commission: float = 0.0
    
    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity - self.commission
    
    @property
    def pnl_percent(self) -> float:
        return (self.exit_price - self.entry_price) / self.entry_price * 100
    
    @property
    def duration(self) -> pd.Timedelta:
        return self.exit_time - self.entry_time
    
    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


class Portfolio:
    def __init__(self, initial_capital: float = 100000.0, commission_per_share: float = 0.005):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_per_share = commission_per_share
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
    @property
    def total_market_value(self) -> float:
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_equity(self) -> float:
        return self.cash + self.total_market_value
    
    @property
    def total_unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_realized_pnl(self) -> float:
        return sum(trade.pnl for trade in self.trades)
    
    def can_afford(self, symbol: str, quantity: int, price: float) -> bool:
        cost = abs(quantity) * price + self.calculate_commission(quantity)
        return cost <= self.cash
    
    def calculate_commission(self, quantity: int) -> float:
        return abs(quantity) * self.commission_per_share
    
    def open_position(self, symbol: str, quantity: int, price: float, timestamp: datetime) -> bool:
        commission = self.calculate_commission(quantity)
        total_cost = abs(quantity) * price + commission
        
        if not self.can_afford(symbol, quantity, price):
            return False
        
        if symbol in self.positions:
            existing_pos = self.positions[symbol]
            if (existing_pos.quantity > 0 and quantity > 0) or (existing_pos.quantity < 0 and quantity < 0):
                # Adding to existing position
                total_quantity = existing_pos.quantity + quantity
                total_cost_existing = abs(existing_pos.quantity) * existing_pos.entry_price
                total_cost_new = abs(quantity) * price
                avg_price = (total_cost_existing + total_cost_new) / abs(total_quantity)
                
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=total_quantity,
                    entry_price=avg_price,
                    entry_time=existing_pos.entry_time,
                    current_price=price
                )
            else:
                # Opposite direction - closing or reversing
                return self._handle_position_close_or_reverse(symbol, quantity, price, timestamp)
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                entry_time=timestamp,
                current_price=price
            )
        
        self.cash -= total_cost
        return True
    
    def _handle_position_close_or_reverse(self, symbol: str, quantity: int, price: float, timestamp: datetime) -> bool:
        existing_pos = self.positions[symbol]
        
        if abs(quantity) <= abs(existing_pos.quantity):
            # Partial or full close
            close_quantity = -min(abs(quantity), abs(existing_pos.quantity)) if existing_pos.quantity > 0 else min(abs(quantity), abs(existing_pos.quantity))
            
            # Record the trade
            commission = self.calculate_commission(abs(close_quantity))
            trade = Trade(
                symbol=symbol,
                quantity=close_quantity,
                entry_price=existing_pos.entry_price,
                exit_price=price,
                entry_time=existing_pos.entry_time,
                exit_time=timestamp,
                commission=commission
            )
            self.trades.append(trade)
            
            # Update cash
            proceeds = abs(close_quantity) * price - commission
            self.cash += proceeds
            
            # Update position
            remaining_quantity = existing_pos.quantity + close_quantity
            if remaining_quantity == 0:
                del self.positions[symbol]
            else:
                self.positions[symbol].quantity = remaining_quantity
        else:
            # Full close and reverse
            close_quantity = -existing_pos.quantity
            reverse_quantity = quantity + close_quantity
            
            # Close existing position
            commission_close = self.calculate_commission(abs(close_quantity))
            trade = Trade(
                symbol=symbol,
                quantity=close_quantity,
                entry_price=existing_pos.entry_price,
                exit_price=price,
                entry_time=existing_pos.entry_time,
                exit_time=timestamp,
                commission=commission_close
            )
            self.trades.append(trade)
            
            proceeds = abs(close_quantity) * price - commission_close
            self.cash += proceeds
            
            # Open new position in opposite direction
            commission_new = self.calculate_commission(abs(reverse_quantity))
            cost_new = abs(reverse_quantity) * price + commission_new
            
            if cost_new <= self.cash:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=reverse_quantity,
                    entry_price=price,
                    entry_time=timestamp,
                    current_price=price
                )
                self.cash -= cost_new
            else:
                del self.positions[symbol]
                return False
        
        return True
    
    def close_position(self, symbol: str, price: float, timestamp: datetime) -> bool:
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        close_quantity = -position.quantity
        
        return self.open_position(symbol, close_quantity, price, timestamp)
    
    def update_prices(self, prices: Dict[str, float], timestamp: datetime):
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
        
        # Record equity curve point
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.total_equity,
            'cash': self.cash,
            'market_value': self.total_market_value,
            'unrealized_pnl': self.total_unrealized_pnl,
            'realized_pnl': self.total_realized_pnl
        })
    
    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)
    
    def is_flat(self, symbol: str) -> bool:
        return symbol not in self.positions
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve)
    
    def get_trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'symbol': trade.symbol,
                'quantity': trade.quantity,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'duration': trade.duration,
                'commission': trade.commission,
                'is_winner': trade.is_winner
            }
            for trade in self.trades
        ])