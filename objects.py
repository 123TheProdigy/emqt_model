from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import enum
from scipy.stats import norm

Time = int
Symbol = str
Product = str
Position = int
UserId = int

## Net position limit mechanism may be useful 

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


class asset_dynamics():
    """
    Class to generate true value of asset prices 
    """
    def __init__(self, p_jump : float, sigma : float, init_price: float):
        self.p_jump = p_jump
        self.std = sigma
        self.p0 = init_price

        self.dynamics = None

    def simulate(self, tmax : int):
        """
        Simulate a price path
        """
        data = np.zeros(tmax) 
        data[0] = self.p0 ## initial value
        data = pd.DataFrame(data)
        
        # initial value is not a jump. It is then random according to p_jump
        jumps = [0] + list(np.random.choice([1, 0], size=tmax-1, p=[self.p_jump, 1-self.p_jump]))
    
        data["jumps"] = jumps
        data["amp"] = np.random.normal(0, self.std, tmax)
        data["reals"] = data.apply(lambda se: se[0] + se["jumps"]*se["amp"], axis=1) # add amplitude only where jumps occurs
        data["price"] = data.reals.cumsum(0)

        ## object now has a "current" dynamics
        self.dynamics = data

    def price(self, tmax  : int) -> pd.Series:
        """
        Return the simulated price 
        """
        if self.dynamics is None:
            self.simulate(tmax=tmax)["price"]
            return self.dynamics["price"]
        elif tmax == len(self.dynamics):
            return self.dynamics["price"]
        else: 
            self.simulate(tmax=tmax)
            return self.dynamics["price"]
        

class Vi_prior():
    """
    This class generates the prior and posterior probabilities of V at time i. The posteriors represent P(V = Vi | Buy/Sell), while the
    priors represent P(V = Vi). It is initialized with an initial true price and spans [v0 - 4*sigma, v0 + 4*sigma - 1], which contains
    the priors. The i-th value inside this vector is the CDF of N(0, sigma) based on i. 
    """

    def __init__(self, sigma: float, centered_at: float):
        """
        sigma: the std of jump 
        centered_at: value around which the discrete vector of probabilities is centered (can be initialized to true value at t=0)
        multiplier: setting the length of vector values according to len=2*4*sigma_price; will allow for more granularity of the discretization
                    of the asset's value space. 
        """

        self.sigma = sigma
        self.center = centered_at
        self.multiplier = 1 # (400/(2 * 4 * sigma)) 
        self.vec_v = None
        self.prior_v = None
        self.v_history = []
        self.p_history = []

        self.compute_vec_v()
        self.compute_prior_v()


    def reset(self, centered_at:Optional[float]=None):
        """
        Reset the prior distribution
        """
        if centered_at is not None:
            self.center = centered_at

        self.compute_vec_v() 
        self.compute_prior_v()
    
    
    def compute_vec_v(self):
        """
        Create vector of range of possible values the asset may have. The 4 used in the calculations represents the number of std
        """
        vec_v = []
        for i in range(int(2 * 4 * self.sigma * self.multiplier + 1)):
            vec_v.append(self.center - 4 * self.sigma + (i / self.multiplier))
        
        self.vec_v = vec_v
        self.v_history.append(vec_v)


    def compute_prior_v(self):
        """
        Create vector of prior probabilities P(V=Vi) from CDF of gaussian with mean 0, std sigma
        """
        prior_v = []
        for i in range(int(2 * 4 * self.sigma * self.multiplier + 1)):
            prior_v.append(norm.cdf(x = -4 * self.sigma + (i+1)/self.multiplier, scale = self.sigma) 
                           - norm.cdf(x = -4 * self.sigma + i/self.multiplier, scale = self.sigma))
        
        self.prior_v = prior_v
        self.p_history.append(prior_v)
    
    def compute_posterior(self, order_type : int, Pbuy : float, Psell : float, Pno : float, Pa : float, Pb : float, 
                          alpha : float, eta : float, sigma_w : float) -> list:
        """
        Compute the posterior probability of P(V=Vi | Order), which represents the updated beliefs about the asset's true value
        given a new trade has occurred. 

        order_type: 1, -1 or 0 respectively for buy, sell, or no order (no order also contains information)
        Pbuy: prior proba of receiving a buy order
        Psell: prior proba of receiving a sell order
        Pno: prior proba of receiving no order
        Pa: ask price of the transaction
        Pb: bid price of the transaction
        alpha: proportion of informed trader (perfectly informed or noisy informed)
        eta: proba of buy/sell order for uninformed trader (again, the market maker is aware of the probabilstic structure 
                of trading agents)
        sigma_w: std of noise distribution (gaussian) of noisy informed traders

        **NOTE: order_type should probably depend on a Trade object; there is high probability that at each iteration, at least one of 
                the agents places a trade. Should investigate what happens here if we do not have any no orders. But given this mainly
                concerns the informed traders, it should probably matter a decent amount 
                But we will likely kill the "no_order" component of this model
        """
        assert Pb < Pa, "ERROR: ask price is below bid price"

        post = []
        if order_type == 1:
            for i, v in enumerate(self.vec_v):
                post.append(self.prior_v[i] * ((1 - alpha) * eta + alpha * (1 - norm.cdf(x = Pa-v, scale = sigma_w))))
            post = np.array(post)/Pbuy
        elif order_type == -1:
            for i, v in enumerate(self.vec_v):
                post.append(self.prior_v[i] * ((1 - alpha) * eta + alpha * norm.cdf(x = Pb-v, scale = sigma_w)))
            post = np.array(post)/Psell
        else:
            for i, v in enumerate(self.vec_v):
                post.append(self.prior_v[i] * ((1- 2*eta) * (1 - alpha) + alpha * (norm.cdf(x = Pa-v, scale = sigma_w) - 
                                                                                   norm.cdf(x = Pb-v, scale = sigma_w))))
            post = np.array(post)/Pno

        self.prior_v = post
        self.p_history.append(post)
        self.v_history.append(self.vec_v)

        return post

class God():
    """
    This class will be used to create the true asset prices and will also contain all the functions to calculate Bayesian priors/posteriors
    as according to the Glosten-Milgrom-Das mode.
    A key idea of this model is that the true asset price is pre-generated, and infused with random jumps to simulate shocks. 
    There are 4 main agent types in this model: market maker, informed trader, noisy informed trader, noise trader
    At time 0, the market maker and informed trader will know the true value of the asset. The noisy informed trader will know the true
    value + some noise from a Gaussian(0, sigma_w^2).
    After the initial time frame, only the informed trader will continue to receive the true value of the asset. The noisy informed trader
    will continue to receive its noised signal. The market maker must infer what the true value is based upon the placed trades. It will then
    create an order book on the level of the best ask and best bid. 
    We will extend this to multiple levels by simply providing +- k=3 tick sizes above and below the bid/ask. If time permits, we will add
    in the liquidity providers from Jericevich et al., which are also suited for dynamically creating synthetic order books. 

    After calculating functions such as Pb, Pa, P_buy, P_sell, P_no_order, we can compute an expected true value for the market maker as well
    as update the posterior distribution. These would all be things done in each iteration. The first thing that should be done is 
    initialize the prior. This prior should be a separate class. 

    The main variables:
        tmax: duration of simulation
        sigma: std of jump distribution (normal)
        jump_prob: probability of a jump in a given iteration
        alpha: proportion of informed traders
        eta: probability an uninformed trader places a buy/sell order
        sigma_w: std of noisy informed trader (normal)
        V0: true initial value
        extend_spread: amount to 1. add to ask 2. remove from bid (o/w will steer towards zero profit for MM)
        gamma: inventory control coefficient for MM profiteering 
        
        jumps: list of if the asset price jumped on iteration i; generated from asset price creation function
        true_value: list of true values at iteration i; generated from asset price creation function
    
    Variables for asset price creation:
        jump_prob
        sigma
        V0
    """
    def __init__(self, tmax: int, sigma: float, jump_prob: float, alpha: float, eta: float, sigma_w: float, 
                 V0: float, extend_spread: Optional[float] = 0, gamma: Optional[float] = 0):
        self.tmax = tmax
        self.sigma = sigma
        self.jump_prob = jump_prob
        self.alpha = alpha
        self.eta = eta
        self.sigma_w = sigma_w
        self.V0 = V0
        self.extend_spread = extend_spread
        self.gamma = gamma

        self.jumps, self.true_value = self.get_asset_dynamics()

    def get_asset_dynamics(self):
        val = asset_dynamics(p_jump=self.proba_jump, sigma=self.sigma_price, init_price=self.V0)
        val.simulate(tmax=self.tmax)

        return val.dynamics["jumps"].to_list(), val.price(tmax=self.tmax).to_list() 

        
