from mesa import Agent, Model
from mesa.time import RandomActivation
from joblib import Parallel, delayed
import random
from collections import OrderedDict
import numpy as np
from scipy.optimize import fixed_point

from objects import Side, Order, Trade


class BookKeeper(Agent):
    def __init__(self, unique_id, model):
        """
        self.bid_book: ordered dictionary of orders on the bid side of the order book 
        self.ask_book: ordered dictionary of orders on the ask side of the order book 
        both self.bid_book and self.ask_book are represented by (key | value) = (price: int | [volume: int, orders: List[Order]])

        self.market_orders: tracks orders from traders
        self.trades: tracks completed trades 
        """
        super().__init__(unique_id, model)
        self.time = 0
        self.asset_price = 100 # arbitrary asset price for now 
        self.bid_book = OrderedDict()
        self.ask_book = OrderedDict()
        self.market_orders = []
        self.trades = []
        self.order_book = [self.bid_book, self.ask_book]

    def receive_order(self, order: Order):
        """
        Trading agent sends order to the market with this function
        """
        self.market_orders.append(order)
        
        self.orders.append(order)

    def insert_bid(self, order: Order):
        keys_to_move = [k for k in self.bid_book if k < order.price]
        if order.price not in self.bid_book:
            self.bid_book[order.price] = [order.quantity, [order]]
            for key in reversed(keys_to_move):
                self.bid_book.move_to_end(key)
        else:
            self.bid_book[order.price][0] += order.quantity
            self.bid_book[order.price][1].append(order)

    def insert_ask(self, order: Order):
        keys_to_move = [k for k in self.ask_book if k > order.price]
        if order.price not in self.ask_book:
            self.ask_book[order.price] = [order.quantity, [order]]
            for key in reversed(keys_to_move):
                self.ask_book.move_to_end(key, last=False)
        else:
            self.ask_book[order.price][0] += order.quantity
            self.ask_book[order.price][1].append(order)

    def send_limit_order(self, order: Order):
        """
        Receiving quote from MM 
        """
        if order.side == Side.BUY:
            self.insert_bid(order)
        else:
            self.insert_ask(order)

    def clear_limit_book(self):
        self.bid_book = OrderedDict()
        self.ask_book = OrderedDict()

    def process_market_orders(self):
        """
        Process all orders at end of iteration
        """
        best_ask = list(self.ask_book.keys())[0]
        best_bid = list(self.bid_book.keys())[0]

        ## no concurrency yet 
        for order in self.market_orders:
            if order.side == Side.BUY:
                while order.quantity > 0 and self.ask_book:
                    best_ask = next(iter(self.ask_book.keys()))
                    ask_quantity, ask_orders = self.ask_book[best_ask]
                    vol = min(ask_quantity, order.quantity)

                    self.trades.append(Trade(best_ask, vol, order.agent_id, "market_maker", self.time))
                    ## need to figure out sending order filled message to agents, although ig doesn't really matter with just one MM 


                    order.quantity -= vol
                    self.ask_book[best_ask][0] -= vol

                    if self.ask_book[best_ask][0] == 0:
                        del self.ask_book[best_ask]

            elif order.side == Side.SELL:
                while order.quantity > 0 and self.bid_book:
                    best_bid = next(iter(self.bid_book.keys()))
                    bid_quantity, bid_orders = self.bid_book[best_bid]
                    vol = min(bid_quantity, order.quantity)

                    self.trades.append(Trade(best_bid, vol, order.agend_id, "market_maker", self.time))
                    ## need to figure out sending order filled message to agents

                    order.quantity -= vol
                    self.bid_book[best_bid][0] -= vol

                    if self.bid_book[best_bid][0] <= 0:
                        del self.bid_book[best_bid]

        ## order clearing?
        for order in self.market_orders:
            agent = next(agent for agent in self.schedule.agents if agent.unique_id == order.agent_id)
            agent.order_failed(order) ## notify agent that their order failed 
            self.market_orders.clear()
            agent.order_failed(order)  ## notify agent that their order failed
        self.orders.clear()

    def step(self):
        # clears orders that are able to be fulfilled?
        pass


class MarketMaker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.price = 100.0
        self.id = unique_id
        
        self.pnl = 0
        self.net_position = 0
        self.best_bids = []
        self.best_asks = []

        self.curr_bid = 0
        self.bid_vol = 0
        self.curr_ask = 0
        self.ask_vol = 0

    def send_limit_order(self, order: Order):
        if order.side == Side.BUY:
            self.insert_bid(order)
        else:
            self.insert_ask(order)  

    def receive_quote(self, quote: tuple):
        """
        The quote will be a tuple in the form (bid, bid_volume, ask, ask_volume)
        """
        self.curr_bid, self.bid_vol, self.curr_ask, self.ask_vol = quote
        self.bid_to_post = Order(self.id, Side.BID, self.curr_bid, self.bid_vol)
        self.ask_to_post = Order(self.id, Side.SELL, self.curr_ask, self.ask_vol)
        self.best_bids.append(self.bid_to_post)
        self.best_asks.append(self.ask_to_post)

    def step(self):
        demand = sum(
            agent.decide_order(self.price) for agent in self.model.schedule.agents if isinstance(agent, TradingAgent))
        self.price += 0.1 * demand


class TradingAgent(Agent):
    """
    General trading agent; there will be several types.
    1. Uninformed noise trader
    2. Noisy informed trader
    3. Fully informed trader
    4. Mean reversion trader
    5. Momentum trader 
    This agent will interact with the market model's limit order book to place trades. 

    self.exchange: the bookkeeper to send trades to / receive messages from
    """

    def __init__(self, unique_id, model, strategy, bookkeeper: BookKeeper):
        super().__init__(unique_id, model)

        self.exchange = bookkeeper

        self.strategy = strategy
        self.position = 0
        self.pnl = 0
        self.posted_bids = []
        self.posted_offers = []
        self.hit_bids = []
        self.hit_asks = []

        self.true_value = 0 # should be "hidden"
        self.failed_orders = []

    def receive_true(self, true_val):
        self.true_value = true_val

    def decide_order(self, price):
        signal = self.strategy.decide_order(price)
        if signal == 0:
            return None
        return Order(self.id, signal, price, 1)

    def order_failed(self, order: Order):
        """
        If the bookkeeper fails to fill an order, we note that it failed 
        """
        self.failed_orders.append(order)
        print(f'Order {order} was not filled.')

    def order_filled(self, trade: Trade):
        """
        If an order is filled by the bookkeeper, then we receive a notification and add it to our trade history 
        """
        side = Side.BUY if trade.buyer == self.unique_id else Side.SELL
        if side == Side.BUY:
            self.hit_bids.append(trade)
            
        else:
            self.hit_asks.append(trade)

    def send_order(self, order):
        """
        Store posted orders and send them to bookkeeper 
        """
        if order.side == Side.BUY:
            self.posted_bids.append(order)
        else:
            self.posted_offers.append(order)

        self.exchange.receive_order(order)
    
    def update_pnl(self, trade : Trade, true_val: int): 
        """
        This will depend a decent amount on control flow because the true value from God needs to be passed in 
        """
        pass

    def step(self):
        order = self.decide_order(self.model.market_maker.price)
        self.model.market_maker.price += order * 0.1


class RandomStrategy:
    def decide_order(self, price):
        return random.choice([-1, 1])
    
class UninformedStrategy:
    def __init__(self, eta: float):
        """
        Eta is the probability of a trade occuring. 
        """
        assert eta <= 0.5, "eta must be less than or equal to 0.5"
        self.eta = eta

    def decide_order(self):
        np.random.choice([1, -1, 0], p=[self.eta, self.eta, 1-2*self.eta])

class NoisyInformedStrategy:
    def __init__(self, sigma_w: float):
        self.sigma_w = sigma_w

    def decide_order(self, bid, ask, true_value):
        noisy_value = true_value + np.random.normal(0, self.sigma_noise)
        if noisy_value > ask:
            return 1
        elif (bid < noisy_value) and (noisy_value < ask):
            return 0
        else:
            return -1

class InformedStrategy:
    def decide_order(self, bid, ask, true_value):
        if true_value > ask:
            return 1
        elif (bid < true_value) and (true_value < ask):
            return 0
        else:
            return -1

class MeanReversionStrategy:
    def __init__(self, window_size=10, threshold=1.0):
        self.window_size = window_size
        self.threshold = threshold

    def calculate_mean(self, price_history):
        if len(price_history) < self.window_size:
            return None

        mean_price = sum(price_history) / len(price_history)
        return mean_price

    def decide_order(self, price_history):
        mean_price = self.calculate_mean(price_history)
        if mean_price is None:
            return 0

        current_price = price_history[-1]
        deviation = current_price - mean_price

        if deviation > self.threshold:
            return -1
        elif deviation < -self.threshold:
            return 1
        else:
            return 0


class NoiseStrategy:
    """
    Noise trader using geometric Brownian motion.

    Either constructor/decide_order to be changed to incorporate past prices
    or fit_parameters must be called before decide_order
    """

    def __init__(self, initial_price=100.0, dt=1):
        self.volatility = None
        self.drift = None
        self.price = initial_price
        self.dt = dt

    def fit_parameters(self, price_history):
        """
        Fit drift and volatility parameters using OLS regression on historical price data.
        """
        returns = np.diff(np.log(price_history))
        X = np.vstack((np.ones_like(returns), np.arange(len(returns)))).T
        y = returns[1:]

        params = np.linalg.lstsq(X, y, rcond=None)[0]

        # Drift is the intercept, volatility is the slope
        self.drift = params[0]
        self.volatility = params[1] / np.sqrt(self.dt)  # Adjust volatility for time step

    def decide_order(self, price_history):
        self.fit_parameters(price_history)
        dW = np.random.normal(loc=0, scale=np.sqrt(self.dt))
        dS = self.drift * self.price * self.dt + self.volatility * self.price * dW
        # self.price += dS

        if dS > 0:
            return 1
        elif dS < 0:
            return -1
        else:
            return 0


class MomentumStrategy:
    """
    Momentum strategy using exponential moving average.
    We could also use other methods like SMA, ROC, etc.
    """

    def __init__(self, window_size=10, alpha=0.1):
        self.window_size = window_size
        self.alpha = alpha  # Smoothing factor for EMA
        self.ema = None

    def calculate_ema(self, price_history):
        if self.ema is None:
            self.ema = price_history[0]
        else:
            self.ema = self.alpha * price_history[-1] + (1 - self.alpha) * self.ema
        return self.ema

    def decide_order(self, price_history):
        if len(price_history) < self.window_size:
            return 0

        ema = self.calculate_ema(price_history)

        # Calculate momentum as the difference between the current price and EMA
        momentum = price_history[-1] - ema

        if momentum > 0:
            return 1
        elif momentum < 0:
            return -1
        else:
            return 0


class MarketModel(Model):
    """
    Market model to represent trading environment. 
    """

    def __init__(self, N):
        """
        Market model to handle agent interactions and other stuff 
        """
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.directory = {}
        self.market_maker = MarketMaker(0, self)
        self.directory[0] = self.market_maker
        self.time = 0
        self.book_keeper = BookKeeper(0, self)

        self.schedule.add(self.book_keeper)
        self.schedule.add(self.market_maker)
        for i in range(1, self.num_agents):
            agent_strategy = random.choice([RandomStrategy(), MeanReversionStrategy()])
            a = TradingAgent(i + 1, self, agent_strategy)
            self.directory[i] = a
            self.schedule.add(a)

        self.running = True

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