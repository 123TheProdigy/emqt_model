from mesa import Agent, Model
from mesa.time import RandomActivation
from joblib import Parallel, delayed
import random
from collections import OrderedDict

from objects import Side, Order, Trade


class BookKeeper(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.order_book = []
        self.trade_history = []

    def record_order(self, order):
        self.order_book.append(order)

    def record_trade(self, trade):
        self.trade_history.append(trade)

    def step(self):
        # clears orders that are able to be fulfilled?
        pass


class MarketMaker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.price = 100.0

    def step(self):
        demand = sum(
            agent.decide_order(self.price) for agent in self.model.schedule.agents if isinstance(agent, TradingAgent))
        self.price += 0.1 * demand


class TradingAgent(Agent):
    """
    General trading agent; there will be two types.
    1. Fundamentalist Trader
    2. Trend Following Trader
    This agent will interact with the market model's limit order book to place trades. 
    """
    def __init__(self, unique_id, model, strategy):
        super().__init__(unique_id, model)
        self.strategy = strategy
        self.position = 0
        self.pnl = 0
        self.posted_bids = []
        self.posted_offers = []
        self.hit_bids = []
        self.hit_asks = [] 

        self.failed_orders = []

    def decide_order(self, price):
        return self.strategy.decide_order(price)
    
    def order_failed(self, order: Order):
        self.failed_orders.append(order)
        print(f'Order {order} was not filled.')

    def order_filled(self, trade: Trade):
        side = Side.BUY if trade.buyer == self.unique_id else Side.SELL
        if side == Side.BUY:
            self.hit_bids.append(trade)
        else:
            self.hit_asks.append(trade)

    def step(self):
        order = self.decide_order(self.model.market_maker.price)
        self.model.market_maker.price += order * 0.1


class RandomStrategy:
    def decide_order(self, price):
        return random.choice([-1, 1])


class MeanReversionStrategy:
    def decide_order(self, price):
        if price > 100:
            return -1
        elif price < 100:
            return 1
        else:
            return random.choice([-1, 1])


class MarketModel(Model):
    """
    Market model to represent trading environment. 
    """
    def __init__(self, N):
        """
        self.bid_book: ordered dictionary of orders on the bid side of the order book 
        self.ask_book: ordered dictionary of orders on the ask side of the order book 
        both self.bid_book and self.ask_book are represented by (key | value) = (price: int | [volume: int, orders: List[Order]])

        self.market_orders: tracks orders from traders
        self.trades: tracks completed trades 
        """
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.market_maker = MarketMaker(0, self)
        self.time = 0
        self.book_keeper = BookKeeper(0, self)

        self.schedule.add(self.book_keeper)
        self.schedule.add(self.market_maker)
        for i in range(self.num_agents):
            agent_strategy = random.choice([RandomStrategy(), MeanReversionStrategy()])
            a = TradingAgent(i + 1, self, agent_strategy)
            self.schedule.add(a)

        self.running = True

        self.bid_book = OrderedDict()
        self.ask_book = OrderedDict()
        self.market_orders = []
        self.trades = []

    def send_order(self, order: Order):
        """
        Agent sends order to the market with this function
        """
        self.orders.append(order)

    def insert_bid(self, order: Order):
        keys_to_move = [k for k in self.bid_book if k < order.price]
        if order.price not in self.bid_book:
            self.bid_book[order.price] = [order.quantity, [order]]
        else:
            self.bid_book[order.price][0] += order.quantity
            self.bid_book[order.price][1].append(order)
        for key in reversed(keys_to_move):
            self.bid_book.move_to_end(key)

    def insert_ask(self, order: Order):
        keys_to_move = [k for k in self.ask_book if k > order.price]
        if order.price not in self.ask_book:
            self.ask_book[order.price] = [order.quantity, [order]]
        else:
            self.ask_book[order.price][0] += order.quantity
            self.ask_book[order.price][1].append(order)
        for key in reversed(keys_to_move):
            self.ask_book.move_to_end(key, last = False)

    def send_limit_order(self, order: Order):
        if order.side == Side.BUY:
            self.insert_bid(order)
        else:
            self.insert_ask(order)

    def process_market_orders(self):
        """
        Process all orders at end of iteration
        """
        best_ask = self.ask_book.keys()[0]
        best_bid = self.bid_book.keys()[0]

        ## no concurrency yet 
        for order in self.market_orders:
            if order.side == Side.BUY:
                while order.quantity > 0 and self.ask_book:
                    best_ask = next(iter(self.ask_book.keys()))
                    ask_quantity, ask_orders = self.ask_book[best_ask]
                    vol = min(ask_quantity, order.quantity)

                    self.trades.append(Trade(best_ask, vol, order.agend_id, "tbd_seller id", self.time))
                    ## need to figure out sending order filled message to agents 

                    order.quantity -= vol
                    self.ask_book[best_ask][0] -= vol
                    
                    if self.ask_book[best_ask][0] == 0:
                        del self.ask_book[best_ask]  

            elif order.side == Side.SELL:
                while order.quantity > 0 and self.bid_book:
                    best_bid = next(iter(self.bid_book.keys()))
                    bid_quantity, bid_orders = self.bid_book[best_bid]
                    vol = min(bid_quantity, order.quantity)

                    self.trades.append(Trade(best_bid, vol, order.agend_id, "tbd_buyer id", self.time))
                    ## need to figure out sending order filled message to agents
                    
                    order.quantity -= vol
                    self.bid_book[best_bid][0] -= vol
                    
                    if self.bid_book[best_bid][0] <= 0:
                        del self.bid_book[best_bid]


        ## order clearing?
        for order in self.market_orders:
            agent = next(agent for agent in self.schedule.agents if agent.unique_id == order.agent_id)
            agent.order_failed(order)
        self.orders.clear()

    def step(self):
        def step(self):
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(agent.step)() for agent in self.schedule.agents
            )
        self.time += 1


def collect_price(model):
    return model.market_maker.price


model = MarketModel(2)
for i in range(100):
    model.step()
    print("Price:", collect_price(model))