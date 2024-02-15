from typing import Dict, List
import enum

Time = int
Symbol = str
Product = str
Position = int
UserId = int

class Side(enum.IntEnum):
    SELL = 0
    BUY = 1

class Order:
    def __init__(self, agent_id: int, side: Side, price: int, quantity: int) -> None:
        """
        Order object; agents (both traders and market makers) should be submitting orders to the market model through this class

        agent_id: id associated with agent
        side: Side.SELL if agent is trying to sell, Side.BUY if agent is trying to buy
        price: market price of order
        quantity: volume to buy/sell
        """
        self.agent_id = agent_id
        self.side = side
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return "(" + self.agent_id + ", " + str(self.price) + ", " + str(self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.agent_id + ", " + str(self.price) + ", " + str(self.quantity) + ")"
    
class Trade:
    """
    Trade object; once an object has been filled, market should record the order with the trade object 
    """
    def __init__(self, price: int, quantity: int, buyer: UserId, seller: UserId, time: Time) -> None:
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.time = time

    def __str__(self) -> str:
        return "(" + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(
            self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(
            self.quantity) + ")"
    

class MMTrade(Trade):
    """
    A subclass of Trade to store more data concerning the order book at trade time 
    """
    def __init__(self, trade : Trade, order_book: List[Dict, Dict]):
        super().__init__(trade.price, trade. quantity, trade.buyer, trade.seller, trade.time)
        self.order_book = order_book 
        
