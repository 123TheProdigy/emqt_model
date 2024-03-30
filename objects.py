from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import enum
from scipy.stats import norm
import math
from scipy.optimize import fixed_point
# from main import NoiseStrategy, InformedStrategy
import matplotlib.pyplot as plt

Time = int
Symbol = str
Product = str
Position = int
UserId = int

## Net position limit mechanism may be useful 

class Side(enum.IntEnum):
    SELL = -1
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
        return "(" + str(self.agent_id) + ", " + str(self.price) + ", " + str(self.quantity) + "," + str(self.side) + ")"

    def __repr__(self) -> str:
        return "(" + str(self.agent_id) + ", " + str(self.price) + ", " + str(self.quantity) + ")"
    
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

    # def __str__(self) -> str:
    #     return "(" + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(
    #         self.quantity) + ")"

    # def __repr__(self) -> str:
    #     return "(" + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(
    #         self.quantity) + ")"
    

class MMTrade(Trade):
    """
    A subclass of Trade to store more data concerning the order book at trade time 
    """
    def __init__(self, trade : Trade, order_book):
        super().__init__(trade.price, trade. quantity, trade.buyer, trade.seller, trade.time)
        self.order_book = order_book


class asset_dynamics():
    """
    Class to generate true value of asset prices 
    """
    def __init__(self, p_jump: float, sigma: float, init_price: float):
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

    def __init__(self, sigma: float, centered_at: float, multiplier: int = 100):
        """
        sigma: the std of jump 
        centered_at: value around which the discrete vector of probabilities is centered (can be initialized to true value at t=0)
        multiplier: setting the length of vector values according to len=2*4*sigma_price; will allow for more granularity of the discretization
                    of the asset's value space. 
        """

        self.sigma = sigma
        self.center = centered_at
        self.multiplier = multiplier # (400/(2 * 4 * sigma)) 
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
        Create vector of possible V_i. The 4 used in the calculations represents the number of std
        """
        # vec_v = []
        # for i in range(int(2 * 4 * self.sigma * self.multiplier + 1)):
            # vec_v.append(self.center - 4 * self.sigma + (i / self.multiplier))
        increments = np.arange(int(2 * 4 * self.sigma * self.multiplier + 1)) / self.multiplier
        vec_v = self.center - 4 * self.sigma + increments
        self.vec_v = vec_v.tolist()
        self.v_history.append(vec_v)


    def compute_prior_v(self):
        """
        Create vector of priors P(V=Vi) from CDF of gaussian with mean 0, std sigma
        """
        # prior_v = []
        # for i in range(int(2 * 4 * self.sigma * self.multiplier + 1)):
            # prior_v.append(norm.cdf(x=-4 * self.sigma + (i + 1) / self.multiplier, scale=self.sigma)
                           # - norm.cdf(x=-4 * self.sigma + i / self.multiplier, scale=self.sigma))
        num_values = int(2 * 4 * self.sigma * self.multiplier + 1)
        indices = np.arange(num_values)
        vec_v = self.center - 4 * self.sigma + indices / self.multiplier
        cdf_values_upper = norm.cdf(-4 * self.sigma + (indices + 1) / self.multiplier, scale=self.sigma)
        cdf_values_lower = norm.cdf(-4 * self.sigma + indices / self.multiplier, scale=self.sigma)
        prior_v = cdf_values_upper - cdf_values_lower

        self.prior_v = prior_v
        # print(f'priors sum to: {np.sum(self.prior_v)}')
        self.p_history.append(prior_v)

    
    def compute_posterior(self, order_type: int, Pbuy: float, Psell: float, Pno: float, Pa: float, Pb: float,
                          alpha: float, beta: float, rho: float, theta: float, mr_true: int, mom_true: int, eta: float,
                          sigma_w: float, scale: float = 1) -> list:
        """
        Compute the posterior probability of P(V=Vi | Order), which represents the updated beliefs about the asset's true value
        given a new trade has occurred.

        To account for volume, we argue that the scaling follows a sigmoid centered at half the order size limit.

        order_type: 1, -1 or 0 respectively for buy, sell, or no order (no order also contains information)
        order_size: size of trade
        Pbuy: prior proba of receiving a buy order
        Psell: prior proba of receiving a sell order
        Pno: prior proba of receiving no order
        Pa: ask price of the transaction
        Pb: bid price of the transaction
        alpha: proportion of informed trader (perfectly informed or noisy informed)
        beta: proportion of uninformed trader
        rho: proportion of mean reverting trader
        theta: proportion of momentum trader
        mr_true: 1 if mean reverting signal positive else 0
        mom_true: 1 if momentum signal positive else 0
        eta: proba of buy/sell order for uninformed trader (again, the market maker is aware of the probabilstic structure
                of trading agents)
        sigma_w: std of noise distribution (gaussian) of noisy informed traders
        scale: scaling factor, from the God class

        **NOTE: order_type should probably depend on a Trade object; there is high probability that at each iteration, at least one of
                the agents places a trade. Should investigate what happens here if we do not have any no orders. But given this mainly
                concerns the informed traders, it should probably matter a decent amount
        """
        assert Pb < Pa, "ERROR: ask price is below bid price"
        # post = []

        # if order_type == 1:
        #     for i, v in enumerate(self.vec_v):
        #         post.append(scale * self.prior_v[i] * (beta * eta + alpha * (1 - norm.cdf(x = Pa-v, scale = sigma_w) + rho * mr_true + theta * mom_true)))
        #     post = np.array(post)/Pbuy

        # elif order_type == -1:
        #     for i, v in enumerate(self.vec_v):
        #         post.append(scale * self.prior_v[i] * (beta * eta + alpha * norm.cdf(x = Pb-v, scale = sigma_w) + rho * mr_true + theta * mom_true))
        #     post = np.array(post)/Psell

        # else:
        #     for i, v in enumerate(self.vec_v):
        #         post.append(self.prior_v[i] * ((1 - 2*eta) * beta + alpha * (norm.cdf(x = Pa-v, scale = sigma_w) - norm.cdf(x = Pb-v, scale = sigma_w)) + rho * mr_true + theta * mom_true))
        #     post = np.array(post)/Pno

        # self.prior_v = post
        # self.p_history.append(post)
        # self.v_history.append(self.vec_v)

        if order_type == 1:

            # for i, v in enumerate(self.vec_v):
                # post.append(self.prior_v[i]*((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-v, scale=sigma_w))))

            # post = np.array(post)/Pbuy
            post = (self.prior_v * ((1 - alpha) * eta + alpha * (1 - norm.cdf(Pa - np.array(self.vec_v), scale=sigma_w)))) / Pbuy


        elif order_type == -1:

            # for i, v in enumerate(self.vec_v):
                # post.append(self.prior_v[i]*((1-alpha)*eta + alpha*norm.cdf(x=Pb-v, scale=sigma_w)))

            # post = np.array(post)/Psell
            post = (self.prior_v * ((1 - alpha) * eta + alpha * norm.cdf(Pb - np.array(self.vec_v), scale=sigma_w))) / Psell


        else:

            # for i, v in enumerate(self.vec_v):
                # post.append(self.prior_v[i]*((1-2*eta)*(1-alpha) + alpha*(norm.cdf(x=Pa-v, scale=sigma_w) - norm.cdf(x=Pb-v, scale=sigma_w))))

            # post = np.array(post)/Pno
            post = (self.prior_v * ((1 - 2 * eta) * beta + alpha * (norm.cdf(Pa - np.array(self.vec_v), scale=sigma_w) - norm.cdf(Pb - np.array(self.vec_v), scale=sigma_w)) + rho * mr_true + theta * mom_true)) / Pno

        self.prior_v = post
        self.p_history.append(post)
        self.v_history.append(self.vec_v)

        # print(f'-------------------------Inside posterior: {post}')

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
        beta: proportion of uninformed traders
        rho: proportion of mean reverting traders
        theta: proportion of momentum traders 
        mr_thresh: deviation threshold for mean reverting trader
        mom_thresh: deviation threshold for momentum trader 
        eta: probability an uninformed trader places a buy/sell order
        sigma_w: std of noisy informed trader (normal)
        V0: true initial value
        directory: map of agents 
        extend_spread: amount to 1. add to ask 2. remove from bid (o/w will steer towards zero profit for MM)
        gamma: inventory control coefficient for MM profiteering 
        
        jumps: list of if the asset price jumped on iteration i; generated from asset price creation function
        true_value: list of true values at iteration i; generated from asset price creation function
    
    Variables for asset price creation:
        jump_prob
        sigma
        V0
    """
    def __init__(self, tmax: int, sigma: float, jump_prob: float, alpha: float, beta: float, rho: float, theta: float, 
                 mr_thresh: float, mom_thresh: float, eta: float, sigma_w: float, V0: float, directory: Dict, extend_spread: Optional[float] = 0.03, 
                 gamma: Optional[float] = 0, mr_window: Optional[int] = 10, mom_window: Optional[int] = 10):
        self.tmax = tmax
        self.sigma = sigma
        self.jump_prob = jump_prob
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.theta = theta
        self.mr_thresh = mr_thresh
        self.mom_thresh = mom_thresh
        self.eta = eta
        self.sigma_w = sigma_w
        self.V0 = V0
        self.extend_spread = extend_spread
        self.gamma = gamma
        self.mr_window = mr_window
        self.mom_window = mom_window
        self.multiplier = 400
        self.eps = 1e-1
        self.directory = directory

        self.jumps, self.true_value = self.get_asset_dynamics()
        # print(self.true_value)
        self.midpoint = self.true_value[0] # initialize LOB midpoint as true price for iteration 1
        self.ptd = [] # prices to date, treat as a stack 
        self.ema = 0 # for momentum strat 
        self.i = 0 # ITERATION # 
        self.jump_here = False 

        self.asks = [] 
        self.bids = []
        self.exp_value = []

        self.return_asks = []
        self.return_bids = []
        self.return_exp = []

        self.Pbuy = None
        self.Psell = None
        self.Pnoorder = None

    def increment_time(self):
        self.i += 1
        self.return_asks.append(self.asks[-1])
        self.return_bids.append(self.bids[-1])
        self.return_exp.append(self.exp_value[-1])
        self.jump_here = False

    def get_true(self):
        return self.true_value[self.i]

    def get_asset_dynamics(self):
        val = asset_dynamics(p_jump=self.jump_prob, sigma=self.sigma, init_price=self.V0)
        val.simulate(tmax=self.tmax)

        return val.dynamics["jumps"].to_list(), val.price(tmax=self.tmax).to_list() 
    
    def P_buy(self, Pa: float, alpha: float, beta: float, rho: float, theta: float, mr: int, mom: int, eta: float,
              sigma_w: float,
              vec_v: list, v_prior: list) -> float:
        """
        The a priori probability of a buy order at some ask price Pa

        Pa: ask price to calculate probability of buy order occuring
        vec_v: vector of possible values for V_i
        v_prior: prior probability of V = V_i
        """
        # result = 0
        # for i, v in enumerate(vec_v):
        #     if v <= Pa:
        #         result += (alpha * (1 - norm.cdf(Pa - v, scale = sigma_w)) + (beta) * eta + rho * mr + theta * mom) * v_prior[i]
        #     else:
        #         result += (alpha * norm.cdf(v - Pa, scale = sigma_w) + (beta) * eta + rho * mr + theta * mom) * v_prior[i]

        # result = (1-alpha)*eta
        # for i, v in enumerate(vec_v):
        # result += alpha*(1-norm.cdf(x=Pa-v,scale=sigma_w))*v_prior[i]
        result = (1 - alpha) * eta + alpha * np.sum(
            (1 - norm.cdf(Pa - np.array(vec_v), scale=sigma_w)) * np.array(v_prior))
        return max(1e-10, result)  # result is 0 when all 6 are informed and causes divide by 0 error later


    def P_sell(self, Pb: float, alpha: float, beta: float, rho: float, theta: float, mr: int, mom: int, eta: float, sigma_w: float,
               vec_v: list, v_prior: list) -> float:
        """
        The a priori probability of a sell order at some bid price Pb

        Pb: bid price to calculate probability of sell order occuring
        vec_v: vector of possible values for V_i
        v_prior: prior probability of V = V_i

        got rid of sizing factor from here
        """
        # result = 0
        # for i, v in enumerate(vec_v):
        #     if v >= Pb:
        #         result += (alpha * (1 - norm.cdf(Pb - v, scale = sigma_w)) + (beta) * eta + rho * mr + theta * mom) * v_prior[i]
        #     else:
        #         result += (alpha * norm.cdf(v - Pb, scale=self.sigma_w) + (beta) * eta + rho * mr + theta * mom) * v_prior[i]

        # result = (1-alpha)*eta
        # for i, v in enumerate(vec_v):
            # result += v_prior[i]*norm.cdf(x=Pb-v, scale=sigma_w)*alpha
        result = (1 - alpha) * eta + alpha * np.sum(norm.cdf(Pb - np.array(vec_v), scale=sigma_w) * np.array(v_prior))
        return result


    def P_no(self, Pb: float, Pa: float, alpha: float, beta: float, rho: float, theta: float, mr: int, mom: int, eta: float, sigma_w: float, 
             vec_v: float, v_prior: float):
        """
        The a priori probability of no order being placed at either the bid price Pb or the ask price Pa

        Pb: bid price 
        Pa: ask price
        vec_v: vector of possible values for V_i
        v_prior: prior probability of V = V_i
        """
        assert Pa > Pb, "something went wrong, your ask is lower than your bid"
        prob = (beta) * (1 - 2*eta)
        mr = 1 - mr
        mom = 1 - mom

        for i, v in enumerate(vec_v):
            prob += v_prior[i] * alpha * (norm.cdf(x = Pa - v, scale = sigma_w) - norm.cdf(x = Pb - v, scale = sigma_w))

        return prob


    def Pb_fp(self, Pb: float, alpha: float, beta: float, rho: float, theta: float, mr: int, mom: int, eta: float,
              sigma_w: float,
              vec_v: float, v_prior: float):
        """
        Fixed point equations to be solved using fixed point iteration

        Pb: bid price
        vec_v: vector of possible values for V_i
        v_prior: prior probability of V = V_i
        """
        assert all(v >= 0 for v in vec_v), "you got negative prices"
        p_sell = self.P_sell(Pb, alpha, beta, rho, theta, mr, mom, eta, sigma_w, vec_v, v_prior)

        vec_v = np.array(vec_v)
        v_prior = np.array(v_prior)

        # cdf_values_below = norm.cdf(Pb - vec_v, scale = sigma_w)
        # cdf_values_above = 1 - cdf_values_below ## fixed version

        # mask_below, mask_above = vec_v <= Pb, vec_v > Pb
        # expected_value_below = np.sum((beta * eta + alpha * cdf_values_below[mask_below]) * vec_v[mask_below] * v_prior[mask_below] + rho * mr + theta * mom)
        # expected_value_above = np.sum((beta * eta + alpha * cdf_values_above[mask_above]) * vec_v[mask_above] * v_prior[mask_above] + rho * mr + theta * mom)

        # print(f'INSIDE PB_FP THE P_BUY IS {p_sell}, EV BELOW IS {expected_value_below}, EV ABOVE IS {expected_value_above}')
        # result = expected_value_below + expected_value_above

        prior_on_v = pd.DataFrame(data=[vec_v, v_prior]).T.rename(columns={0: "v", 1: "p"})
        result = sum([((1 - alpha) * eta + alpha * norm.cdf(x=Pb - Vi, scale=sigma_w)) * Vi * (
            prior_on_v[prior_on_v["v"] == Vi]["p"].item()) for Vi in vec_v if Vi <= Pb])
        # print(f'initial sum: {result}')
        result += sum([((1 - alpha) * eta + alpha * norm.cdf(x=Pb - Vi, scale=sigma_w)) * Vi * (
            prior_on_v[prior_on_v["v"] == Vi]["p"].item()) for Vi in vec_v if Vi > Pb])

        # print(f'iiiiiiiiii the result is {result}, the psell is {p_sell}')

        return result / p_sell


    def Pa_fp(self, Pa: float, alpha: float, beta: float, rho: float, theta: float, mr: int, mom: int, eta: float,
              sigma_w: float,
              vec_v: float, v_prior: float):
        """
        Fixed point equations to be solved using fixed point iteration

        Pa: ask price
        vec_v: vector of possible values for V_i
        v_prior: prior probability of V = V_i
        """
        assert all(v >= 0 for v in vec_v), "you got negative prices"
        p_buy = self.P_buy(Pa, alpha, beta, rho, theta, mr, mom, eta, sigma_w, vec_v, v_prior)

        vec_v = np.array(vec_v)
        v_prior = np.array(v_prior)

        # cdf_values_below =  norm.cdf(Pa - vec_v, scale = sigma_w)
        # cdf_values_above = 1 - cdf_values_below

        # mask_below, mask_above = vec_v <= Pa, vec_v > Pa
        # expected_value_below = np.sum((beta * eta + alpha * cdf_values_above[mask_below]) * vec_v[mask_below] * v_prior[mask_below] + rho * mr + theta * mom)
        # expected_value_above = np.sum((beta * eta + alpha * cdf_values_below[mask_above]) * vec_v[mask_above] * v_prior[mask_above] + rho * mr + theta * mom)

        # print(f'INSIDE PA_FP THE P_BUY IS {p_buy}, EV BELOW IS {expected_value_below}, EV ABOVE IS {expected_value_above}')
        # result = expected_value_below + expected_value_above

        prior_on_v = pd.DataFrame(data=[vec_v, v_prior]).T.rename(columns={0: "v", 1: "p"})
        result = sum([((1 - alpha) * eta + alpha * (1 - norm.cdf(x=Pa - Vi, scale=sigma_w))) * Vi * (
            prior_on_v[prior_on_v["v"] == Vi]["p"].item()) for Vi in vec_v if Vi <= Pa])
        result += sum([((1 - alpha) * eta + alpha * (1 - norm.cdf(x=Pa - Vi, scale=sigma_w))) * Vi * (
            prior_on_v[prior_on_v["v"] == Vi]["p"].item()) for Vi in vec_v if Vi > Pa])

        return result / p_buy


    def sizing_factor(self, size, scale=.1):
        a = 1 - self.alpha
        b = self.eta
        exponent = -scale * 3.36358566 * (a ** 2 / (b ** .25 + .25)) * (size - 10)  # center in middle
        return (1 / (1 + math.exp(exponent))) + 0.5


    def sizing_factor(self, size, scale = .1):
        a = 1 - self.alpha
        b = self.eta
        exponent = -scale * 3.36358566 * (a**2/(b**.25 + .25)) * (size - 10) # center in middle
        return (1/(1+math.exp(exponent))) + 0.5


    def compute_exp_true_value(self, Pb: float, Pa: float, psell: float, pbuy: float, vec_v: list, v_prior: list, alpha: float, eta: float, sigma_w: float) -> float:
        """
        Compute the expected value of the asset; only use informed traders? 

        Pb: bid price
        Pa: ask price
        psell: prior probability of a sell order
        pbuy: prior probability of a buy order
        vec_v: vector of V_i values
        v_prior: prior probability of V = V_i
        alpha: proportion of informed traders
        beta: proportion of uninformed traders
        rho: proportion of mean reversion traders
        theta: proportion of momentum traders
        eta: uninformed trader buy/sell probability
        sigma_w: std of noise for noisy informed traders
        """

        exp = Pa*psell + Pb*pbuy 
        for i, v in enumerate(vec_v):
            exp += v*v_prior[i]*alpha*(norm.cdf(x=Pa-v, scale = sigma_w) - norm.cdf(x=Pb-v, scale = sigma_w)) # effectively the prob true value is between the bid/ask
        
        return exp
    
    def comp_mr_indicator(self):
        if len(self.ptd) < self.mr_window:
            return 0

        mean_price = sum(self.ptd[:self.mr_window]) / self.mr_window
        deviation = self.ptd[0] - mean_price
        if abs(deviation) > self.mr_thresh:
            return 1
        return 0
    
    def comp_mom_indicator(self, alpha: float = 0.1):
        if len(self.ptd) < self.mom_window:
            return 0

        if self.ema is None:
            self.ema = sum(self.ptd[-self.mom_window:]) / self.mom_window
        else:
            self.ema = alpha * self.ptd[0] + (1 - alpha) * self.ema
        
        momentum = self.ptd[0] - self.ema
        if abs(momentum) > self.mom_thresh:
            return 1
        return 0
    
    def send_info(self, bid_price, bid_volume, ask_price, ask_volume, true_val):
        """
        Sends information to market agents by iterating through hashmap 

        directory: hashmap of all agents and their respective strategies 
        """
        mm_quote = (round(bid_price, 2), bid_volume, round(ask_price, 2), ask_volume)
        informed_val = true_val
        for id, agent in self.directory.items():
            if id == 0:
                agent.receive_quote(mm_quote)
            else:
                agent.receive_true(true_val)

    def receive_trade(self, interaction_result):
        """
        Receive trade for updating posterior 

        interaction_result: tuple(Bool, Trade), bool is true if an order was placed, false if no order was placed 
                            and Order is none if no order placed
        """
        trade_type = 0
        if interaction_result[0] == False:
            trade_type = 0
        elif interaction_result[2].side == Side.BUY:
            trade_type = 1
        else:
            trade_type = -1

        self.run_and_advance(trade_type)
    
    def run_and_advance(self, trade = None):
        """
        Essentially taking a time step (such as the market model step function)

        Need to create senders and receivers for messages to MM and traders 
        """
        # print(f"++++++++++trade processing at time {self.i}++++++++++")
        if (self.i == 0):
            self.v_distrib = Vi_prior(sigma = 0.5, centered_at = self.V0, multiplier = self.multiplier)
            # print(f'initial prior distribution is: {self.v_distrib.vec_v[-1]} with probs {self.v_distrib.v_history[-1]}')

        if self.jumps[self.i] == 1 and self.jump_here is False:
            self.v_distrib.reset(centered_at=self.exp_value[-1])
            self.jump_here = True

        self.mr_indicator = self.comp_mr_indicator()
        self.mom_indicator = self.comp_mom_indicator()

        # self.pnl = [] # shouldn't need, MM is not God
        # self.inventory = [0] # shouldn't need, MM is not God 

        if self.jumps[self.i] == 1:
            self.v_distrib.reset(centered_at=self.exp_value[-1])

        curr_ask = fixed_point(self.Pa_fp, self.true_value[self.i], args=(self.alpha, self.beta, self.rho, 
                                                                          self.theta, self.mr_indicator, self.mom_indicator, 
                                                                          self.eta, self.sigma_w, self.v_distrib.vec_v, 
                                                                          self.v_distrib.prior_v), xtol=1e-2, maxiter=500, method='del2').item()
        # curr_ask += -self.gamma * self.inventory[-1]
        # print(f'immediately returned ask is {curr_ask}')
        curr_ask += self.extend_spread ## extend spread
        curr_ask = round(curr_ask, 2)
        self.asks.append(curr_ask)

        curr_bid = fixed_point(self.Pb_fp, self.true_value[self.i], args=(self.alpha, self.beta, self.rho, 
                                                                          self.theta, self.mr_indicator, self.mom_indicator, 
                                                                          self.eta, self.sigma_w, self.v_distrib.vec_v, 
                                                                          self.v_distrib.prior_v), xtol=1e-2, maxiter=500, method='del2').item()
        # curr_bid += -self.gamma*self.inventory[-1]
        # print(f'immediately returned bid is {curr_bid}')
        curr_bid += -self.extend_spread ## extend spread
        curr_bid = round(curr_bid, 2)
        self.bids.append(curr_bid)

        # print(f'current bid, ask from inside God: {round(curr_bid,2), round(curr_ask, 2)}')

        self.Pbuy = self.P_buy(Pa=self.asks[-1], alpha=self.alpha, beta=self.beta, rho=self.rho, theta=self.theta, 
                          mr=self.mr_indicator, mom=self.mom_indicator, eta=self.eta, sigma_w=self.sigma_w, 
                          vec_v=self.v_distrib.vec_v, v_prior=self.v_distrib.prior_v)
        assert self.Pbuy>0-self.eps and self.Pbuy<1+self.eps, "Pbuy not between 0 and 1"
        
        # print(f'bid buy prob: {self.Pbuy}')

        self.Psell = self.P_sell(Pb=self.bids[-1], alpha=self.alpha,  beta=self.beta, rho=self.rho, theta=self.theta, 
                            mr=self.mr_indicator, mom=self.mom_indicator, eta = self.eta, sigma_w=self.sigma_w, 
                            vec_v=self.v_distrib.vec_v, v_prior=self.v_distrib.prior_v)
        assert self.Psell>0-self.eps and self.Psell<1+self.eps, "Psell not between 0 and 1"
        
        # print(f'ask sell prob: {self.Pbuy}')

        # print(f'curr bid: {self.bids[-1]}, curr ask: {self.asks[-1]}, vec_v = {self.v_distrib.vec_v}, prior = {self.v_distrib.prior_v}')

        self.Pnoorder = self.P_no(Pb=self.bids[-1], Pa=self.asks[-1], alpha=self.alpha,  beta=self.beta, rho=self.rho, 
                                   theta=self.theta, mr=self.mr_indicator, mom=self.mom_indicator, eta=self.eta, 
                                   sigma_w=self.sigma_w, vec_v=self.v_distrib.vec_v, v_prior=self.v_distrib.prior_v)
        assert self.Pnoorder>0-self.eps and self.Pnoorder<1+self.eps, "P_noorder not between 0 and 1"
        
        # print(f'00000000000000000000000000 pbuy: {self.Pbuy}, psell: {self.Psell}, pno: {self.Pnoorder}')

        assert self.Psell+self.Pbuy+self.Pnoorder>0-self.eps and self.Pbuy + self.Psell + self.Pnoorder<1+self.eps

        self.exp_value.append(self.compute_exp_true_value(Pb=self.bids[-1], Pa=self.asks[-1], psell=self.Psell, pbuy=self.Pbuy, 
                                                          vec_v=self.v_distrib.vec_v, v_prior=self.v_distrib.prior_v, 
                                                          alpha=self.alpha, eta=self.eta, sigma_w=self.sigma_w))
    
        # print(f'----------------expected value is {self.exp_value[-1]}, true value is {self.true_value[self.i]}')

        if trade is not None:
            self.v_distrib.compute_posterior(trade, Pbuy=self.Pbuy, Psell=self.Psell, Pno=self.Pnoorder, Pa=self.asks[-1], Pb=self.bids[-1], 
                                            alpha=self.alpha, beta=self.beta, rho=self.rho, theta=self.theta, 
                                            mr_true=self.mr_indicator, mom_true=self.mom_indicator, eta=self.eta, 
                                            sigma_w=self.sigma_w)
            # print(f'====================posterior is: {np.abs(sum(self.v_distrib.prior_v) - 1)}')
            # assert np.abs(sum(self.v_distrib.prior_v) - 1) < self.eps, "posterior prob is not normalized"


        self.send_info(curr_bid, 10, curr_ask, 10, self.true_value[self.i])

    def return_data(self):
        """
        Returns a tuple of lists containing data
        """
        return self.true_value, self.return_exp, self.return_bids, self.return_asks

    def show_result(self, 
                    figsize:Optional[tuple]=(10,9),
                    dpi:Optional[int]=100, 
                    same_y_axis_above:Optional[bool]=True):
        '''
        This methods shows plots of:
            - the true value and the expected value over time
            - bid/ask prices over time
            - spread over time
            - MM true value distribution at 3 different iterations
        '''

        fig, ax = plt.subplots(2,2, figsize=(figsize[0],figsize[1]), dpi=dpi)

        bids=np.array(self.bids)
        asks=np.array(self.asks)
        

        ax[0, 1].plot(self.bids, label="bid price", alpha=0.8)
        ax[0, 1].plot(self.asks, label="ask price", alpha=0.8)
        ax[0,1].legend()
        ax[0,1].set_xlabel("time t")

        ax[0,0].plot(self.true_value, label="True value", alpha=0.8)
        ax[0,0].set_ylabel("asset value")
        ax[0,0].plot(self.exp_value, label="Exp. value", alpha=0.8)
        ax[0,0].legend()
        if same_y_axis_above:
            ax[0,0].set_ylim(ax[0,1].get_ylim())
        ax[0,0].set_xlabel("time t")

        # ax[1,0].plot((asks-bids)/(0.5*(asks+bids)), label="spread")
        ax[1,0].plot((asks-bids), label="asbolute spread", alpha=0.8)

        ax[1,0].set_xlabel("time t")
        ax[0,1].set_ylabel("bid/ask")
        ax[1,0].set_ylabel("absolute spread")

        #snapshots_i = (int(self.tmax/3), int(2*self.tmax/3), int(3*self.tmax/3-2))
        snapshots_i = [3,6,9]
        for snap in snapshots_i:
            ax[1,1].plot(self.v_distrib.v_history[snap], self.v_distrib.p_history[snap], label=f"iter: {snap}", alpha=0.8)
        ax[1,1].legend()
        ax[1,1].set_xlabel("true values")
        ax[1,1].set_ylabel("count (normalized)")

        fig.tight_layout()

        plt.show()