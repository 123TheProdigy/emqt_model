import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from typing import Optional
from scipy.optimize import fixed_point


class Vi_prior():
    '''
    This class represents the prior probability on the true value V
    that the market maker keeps updated at all times
    It is initialized as a gaussian (discrete vector of probabilities) and
    can be updated at all times with new trade information coming in
    '''

    def __init__(self, sigma_price : float, centered_at : float, multiplier : int, nb_sigma_range : Optional[int] = 4):
        '''
        Args:
            - sigma_price: the std of the asset returns distribution 
            - centered_at: value around which the discrete vector of proba
            is centered (can be initialized to true value at t=0)
            - multiplier: setting the length of vector values according to 
            len=2*4*sigma_price*multiplier
        '''

        self.sigma = sigma_price
        self.center = centered_at
        self.multiplier = multiplier
        self.nb_sigma_range = nb_sigma_range
        self.vec_v = None
        self.prior_v = None
        self.v_history = []
        self.p_history = []
        self.compute_vec_v()
        self.compute_prior_v()


    def reset(self, centered_at:Optional[float]=None):
        '''
        This methods resets the prior distribution 
        Args: 
            - centered_at: if provided, the new distribution will be recented at this value 
        '''

        if centered_at is not None:
            self.center = centered_at
        
        self.compute_vec_v(update_history=True) ## those 2 methods store the new vectors 
        self.compute_prior_v(update_history=True)
    
    
    def compute_vec_v(self, update_history:Optional[bool]=True):
        '''
        This method creates the vector of possible values of v
        If the center changed: need to update it beforehand
        
        Args: 
            - update_history: if True (default), the vector is added to a list with previous 
            vectors to keep track of them 
        '''

        # vec_v = []
        # for i in range(int(2*self.nb_sigma_range*self.sigma*self.multiplier+1)):
            # vec_v.append(self.center-self.nb_sigma_range*self.sigma+i/self.multiplier)
        # self.vec_v = vec_v
        nb_values = int(2 * self.nb_sigma_range * self.sigma * self.multiplier + 1)
        increments = np.arange(nb_values) / self.multiplier
        vec_v = self.center - self.nb_sigma_range * self.sigma + increments
        self.vec_v = vec_v.tolist()

        if update_history:
            self.v_history.append(vec_v)


    def compute_prior_v(self, update_history:Optional[bool]=True):
        '''
        This method creates the vector of probas for v

        Args:
            - update_history: if True (default), the vector is added to a list with previous 
            vectors to keep track of them 
        '''

        # prior_v = []
        # for i in range(int(2*self.nb_sigma_range*self.sigma*self.multiplier+1)):
            # prior_v.append(norm.cdf(x=-self.nb_sigma_range*self.sigma+(i+1)/self.multiplier, scale=self.sigma) - norm.cdf(x=-self.nb_sigma_range*self.sigma+i/self.multiplier, scale=self.sigma))
        
        # self.prior_v = prior_v
        nb_values = int(2 * self.nb_sigma_range * self.sigma * self.multiplier + 1)
        increments = np.arange(nb_values) / self.multiplier
        x_values = -self.nb_sigma_range * self.sigma + increments
        prior_v = (norm.cdf(x=-self.nb_sigma_range * self.sigma + (increments + 1) / self.multiplier,scale=self.sigma)
                   - norm.cdf(x=-self.nb_sigma_range * self.sigma + increments / self.multiplier, scale=self.sigma))
        self.prior_v = prior_v.tolist()

        if update_history:
            self.p_history.append(prior_v)


    
    def compute_posterior(self, 
                            order_type : int, 
                            Pbuy : float, 
                            Psell : float, 
                            Pno : float, 
                            Pa : float, 
                            Pb : float, 
                            alpha : float, 
                            eta : float, 
                            sigma_w : float, 
                            update_prior : Optional[bool]=True, 
                            update_v_vec : Optional[bool]=True) -> list:
                    
        '''
        This methods computes the posterior or P(V=Vi) when trade info is received
        It will update the Market Maker belief of the true value
        Args:
            - order_type: 1, -1 or 0 respectively for buy, sell, or no order 
            (no order also contains information)
            - Pbuy: prior proba of receiving a buy order
            - Psell: prior proba of receiving a sell order
            - Pno: prior proba of receiving no order
            - Pa: ask price of the transaction
            - Pb: bid price of the transaction
            - alpha: proportion of informed trader (perfectly informed or noisy informed)
            - eta: proba of buy/sell order for uninformed trader (again, the market maker is 
            aware of the probabilstic structure of trading agents)
            - sigma_w: std of noise distribution (gaussian) of noisy informed traders
            - update_prior: if True (default), will update the prior (as well as prior history) 
        Output:
            will return the posterior distribution 
        '''

        assert Pb < Pa, "ask price is below bid price"

        post = []

        if order_type == 1:

            # for i, v in enumerate(self.vec_v):
                # post.append(self.prior_v[i]*((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-v, scale=sigma_w))))

            # post = np.array(post)/Pbuy
            post = self.prior_v * ((1 - alpha) * eta + alpha * (1 - norm.cdf(Pa - self.vec_v, scale=sigma_w)))
            post /= Pbuy
        
        elif order_type == -1:

            # for i, v in enumerate(self.vec_v):
                # post.append(self.prior_v[i]*((1-alpha)*eta + alpha*norm.cdf(x=Pb-v, scale=sigma_w)))

            # post = np.array(post)/Psell
            post = self.prior_v * ((1 - alpha) * eta + alpha * norm.cdf(Pb - self.vec_v, scale=sigma_w))
            post /= Psell

        else:

            # for i, v in enumerate(self.vec_v):
                # post.append(self.prior_v[i]*((1-2*eta)*(1-alpha) + alpha*(norm.cdf(x=Pa-v, scale=sigma_w) - norm.cdf(x=Pb-v, scale=sigma_w))))

            # post = np.array(post)/Pno
            post = self.prior_v * ((1 - 2 * eta) * (1 - alpha) + alpha * (
                        norm.cdf(Pa - self.vec_v, scale=sigma_w) - norm.cdf(Pb - self.vec_v, scale=sigma_w)))
            post /= Pno
            
        if update_prior:
            self.prior_v = post
            self.p_history.append(post)
        if update_v_vec:
            self.v_history.append(self.vec_v)

        return post

##   Prior proba of selling

def P_sell(Pb : float, 
            alpha : float, 
            eta : float, 
            sigma_w : float, 
            vec_v:Optional[list], 
            v_prior:Optional[list], 
            known_value:Optional[float]=None) -> float:
    '''
    This is the prior proba of a selling order arriving
    Args:
        - Pb: the bid price to solve (FP eq)
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi

    Returns: 
        - prior proba of the MM receiving a sell order
    '''
    if (known_value is not None):
        assert known_value>0, "known value is negative, cannot compute P_sell"

        if known_value < Pb:
            result = alpha*norm.cdf(x=Pb-known_value, scale=sigma_w) + (1-alpha)*eta
        else:
            result = alpha*(1-norm.cdf(x=known_value-Pb, scale=sigma_w)) + (1-alpha)*eta

    else:

        result = (1-alpha)*eta
        # for i, v in enumerate(vec_v):
            # result += v_prior[i]*norm.cdf(x=Pb-v, scale=sigma_w)*alpha
        result += np.sum(v_prior * norm.cdf(Pb - np.array(vec_v), scale=sigma_w) * alpha)
    return result

## fixed point equation for Bid price

def Pb_fp(Pb : float, 
            alpha : float, 
            eta : float, 
            sigma_w : float, 
            vec_v:Optional[list], 
            v_prior:Optional[list], 
            known_value:Optional[float]=None) -> float:
    '''
    This is the fixed point equation for the bid price Pb
    Args:
        - Pb: the bid price to solve (FP eq)
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi
    Ooutput: 
        - bid price sollution to fixed point equation 
    '''

    ## compute prior proba of sell order
    psell = P_sell(Pb=Pb, alpha=alpha, eta=eta, sigma_w=sigma_w, vec_v=vec_v, v_prior=v_prior, known_value=known_value)
    
    if known_value is not None:
        assert known_value > 0, "known value is negative, cannot compute P_sell"
        if known_value <= Pb:
            result = ((1-alpha)*eta + alpha*norm.cdf(x=Pb-known_value, scale=sigma_w))*known_value
        else:
            result = ((1-alpha)*eta + alpha*(1-norm.cdf(x=known_value-Pb, scale=sigma_w)))*known_value
            
    else:

        prior_on_v = pd.DataFrame(data=[vec_v, v_prior]).T.rename(columns={0:"v", 1:"p"})

        result = sum([((1-alpha)*eta + alpha*norm.cdf(x=Pb-Vi, scale=sigma_w))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi <= Pb])

        result += sum([((1-alpha)*eta + alpha*norm.cdf(x=Pb-Vi, scale=sigma_w))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi > Pb])

    return result/psell



##   Prior proba of buying

def P_buy(Pa : float, 
            alpha : float, 
            eta : float, 
            sigma_w : float, 
            vec_v: list, 
            v_prior: list, 
            known_value:Optional[float]=None) -> float:
    '''
    This is the prior proba of a buying order arriving
    Args:
        - Pa: the ask price to solve (FP eq)
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi
    Output:
        - prior proba of buy order
    '''

    if (known_value is not None):
        assert known_value>0, "known value is negative, cannot compute P_buy"

        if known_value <= Pa:
            result = alpha*(1-norm.cdf(x=Pa-known_value, scale=sigma_w)) + (1-alpha)*eta
        else:
            result = alpha*norm.cdf(x=known_value-Pa, scale=sigma_w) + (1-alpha)*eta

    else:

        result = (1-alpha)*eta
        # for i, v in enumerate(vec_v):
            # result += alpha * (1 - norm.cdf(x=Pa - v, scale=sigma_w)) * v_prior[i]
        result += np.sum(alpha * (1 - norm.cdf(Pa - np.array(vec_v), scale=sigma_w)) * np.array(v_prior))

    return result



##   Prior proba of no order


def P_no_order(Pb : float,
                Pa : float, 
                alpha : float, 
                eta : float, 
                sigma_w : float, 
                vec_v : list, 
                v_prior : list) -> float:
    '''
    This is the prior proba of a no order arriving
    Args:
        - Pb: the bid price 
        - Pa: the ask price 
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi
    Output:
        - prior proba of no order 
    '''

    assert Pa > Pb, "ask is lower than bid"

    prob = (1-alpha)*(1-2*eta) ## part of uninformed traders

    # for i, v in enumerate(vec_v):

        # prob += v_prior[i]*alpha*(norm.cdf(x=Pa-v, scale=sigma_w) - norm.cdf(x=Pb-v, scale=sigma_w))

    prob += np.sum(v_prior * alpha * (norm.cdf(Pa - np.array(vec_v), scale=sigma_w) - norm.cdf(Pb - np.array(vec_v), scale=sigma_w)))

    return prob




## fixed point equation for ask price

def Pa_fp(Pa : float, 
            alpha : float,
            eta : float, 
            sigma_w : float, 
            vec_v: list, 
            v_prior: list, 
            known_value:Optional[float]=None) -> float:
    '''
    This is the fixed point equation for the ask price Pa
    Args:
        - Pa: the ask price to solve (FP eq)
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi

    Output:
        - the ask price solution of the FP equation
    '''
    
    
    # prior proba of buying order arriving
    pbuy = P_buy(Pa=Pa, alpha=alpha, eta=eta, sigma_w=sigma_w, vec_v=vec_v, v_prior=v_prior, known_value=known_value)

    if known_value is not None:
        assert known_value > 0, "known value is negative, cannot compute P_buy"
        if known_value <= Pa:
            result = ((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-known_value, scale=sigma_w)))*known_value
        else:
            result = ((1-alpha)*eta + alpha*norm.cdf(x=known_value-Pa, scale=sigma_w))*known_value
            
    
    else:

        prior_on_v = pd.DataFrame(data=[vec_v, v_prior]).T.rename(columns={0:"v", 1:"p"})

        result = sum([((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-Vi, scale=sigma_w)))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi <= Pa]) 

        result += sum([((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-Vi, scale=sigma_w)))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi > Pa])
                
    return result/pbuy




## expectation of true value

def compute_exp_true_value(Pb : float, 
                            Pa : float, 
                            psell : float, 
                            pbuy : float, 
                            vec_v : list, 
                            v_prior : list, 
                            alpha : float,
                            eta : float,
                            sigma_w : float) -> float:
    '''
    This methods compute the expected value of the asset 
    Args:
        - Pb: bid price
        - Pa: ask price
        - psell: prior proba of sell order
        - pbuy: prior proba of buy oder
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi
        - alpha: proportion of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
    
    Output:
        - expected value
    '''

    exp = Pa*psell + Pb*pbuy ## expected value conditionned on buying order, selling order

    for i, v in enumerate(vec_v):
        ## expected value conditionned on no order arriving (was not already computed)
        exp += v*v_prior[i]*alpha*(norm.cdf(x=Pa-v, scale=sigma_w) - norm.cdf(x=Pb-v, scale=sigma_w))
    
    return exp


class asset_dynamics():
    '''
    This class implements the fake asset price described in Das Paper 
    link: https://cs.gmu.edu/~sanmay/papers/das-qf-rev3.pdf
    
    Args:
        - p_jump: probability of a jump occuring at a each iteration
        - sigma: std of the gaussian distribution representing the amplitude (and direction) of the jump if it occurs
        - init_price: initial price 
    '''

    def __init__(self, 
                    p_jump : float, 
                    sigma : float, 
                    init_price: float):

        self.std = sigma
        self.p0 = init_price
        self.p_jump = p_jump
        self.dynamics = None

    
    def simulate(self, tmax : int):
        '''
        This methods simulates a price path
        Args:
            - tmax: duration of simulation
        '''

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
        '''
        This methods returns the simulated price (re-simulates it with tmax if not the same as current 
        simulation tmax (if any current simulation))
        Args: 
            - tmax: duration of simulation
        Returns: 
            - fake asset price timeseries
        '''

        if self.dynamics is None:
            self.simulate(tmax=tmax)["price"]
            return self.dynamics["price"]
        elif tmax == len(self.dynamics):
            return self.dynamics["price"]
        else: 
            self.simulate(tmax=tmax)
            return self.dynamics["price"]


class GMD_simluation():

    '''This class allows to run a full simulation of a GMD model on an fake asset
        - The asset price is a jump process
        - there are 2 types of traders:
            - uninformed traders
            - noisy informed traders (perfectly informed if the sigma_noise is set to very small values)
        - the market maker knows:
            - the initial asset value
            - the probabilstic nature of trading crowd
            - the *occurence* of jumps in the true asset price

       Params:
            - tmax: duration of simulation
            - sigma_price: std of jumps distribution (normal) (responsible with "multiplier" of simulation duration)
            - proba_jump: probability of a jump at any iteration
            - alpha: proportion of (noisy) informed traders
            - eta: probability of a buy/sell order from an uninformed trader
            - sigma_noise: std of noise distribution (normal) of noisy informed trader
            - V0: initial true value
            - multiplier: parameter setting the discretization qtty (responsible with "sigma_price" of simulation duration)
                - default will lead to precision of cents 
                - higher will leads to finer grid (longer simulation)
            - eps_discrete_error: allowed discretization error
            - extend_spread: amount (in cents) to add to ask and remove from bid to steer free from zero profit condition for MM
            - gamma: inventory control coefficient
        '''


    def __init__(self, 
                    tmax : int, 
                    sigma_price : float, 
                    proba_jump : float, 
                    alpha : float, 
                    eta : float, 
                    sigma_noise : float, 
                    V0 : Optional[float]=100,
                    multiplier : Optional[int]=None, 
                    eps_discrete_error:Optional[float]=1e-4, 
                    extend_spread: Optional[float] = 0,
                    gamma:Optional[float]=0):

        ## atttributes/params of simulation
        self.tmax = tmax
        self.alpha = alpha
        self.sigma_price = sigma_price
        self.proba_jump = proba_jump
        self.eta = eta
        self.sigma_w = sigma_noise
        self.V0 = V0
        self.multiplier = multiplier
        self.eps = eps_discrete_error
        self.run = False
        self.extend_spread = extend_spread
        self.gamma = gamma
        
        ## compute price dynamics
        self.get_asset_dynamics()


    def __str__(self) -> str:
        '''
        method printing summary of simulation params
        '''

        res = f"GMD simulation with following params: \n- {self.tmax} iterations\n"
        res += f"- price with jump probability {self.proba_jump} and amplitude std {self.sigma_price} (current path has {self.jumps.count(1)} jumps)\n"
        res += f"- proportion informed traders is {self.alpha}\n"
        res += f"- noise std of informed traders is {self.sigma_w}\n"
        res += f"- probability of random trade by uninformed traders is {self.eta}\n"
        if self.multiplier is None:
            res += f"- V(t=0)={self.V0}, multiplier is set to default"
        else:
            res += f"- V(t=0)={self.V0}, multiplier is {self.multiplier} so prob density len is {int(self.multiplier*2*self.sigma_price*4)}"
        if not self.run:
            res += "\n       Not run yet       "
        else:
            res += "\n       Already run       "
        return res


    

    def get_asset_dynamics(self):
        '''
        This methods simulates one asset price path and saves it for when the 
        a simulation is run
        '''
        
        val = asset_dynamics(p_jump=self.proba_jump, sigma=self.sigma_price, init_price=self.V0)

        val.simulate(tmax=self.tmax)

        self.jumps = val.dynamics["jumps"].to_list()
        self.true_value = val.price(tmax=self.tmax).to_list() 




    def run_simulation(self):
        '''
        This methods runs one simulation with the given parameters (if already run, this 
        method will overwrite previous results)
        '''

        if self.multiplier is None:
            self.multiplier = int(400/(2*self.sigma_price*4))

        ## initialize prior
        self.v_distrib = Vi_prior(sigma_price=self.sigma_price, centered_at=self.V0, multiplier=self.multiplier)

        self.asks = []
        self.bids = []
        self.exp_value = []
        self.pnl = []
        self.inventory = [0]

        ## initialize trading crowd and traders order
        self.u_trader = uninformed_trader(trade_prob=self.eta)
        self.i_trader = noisy_informed_trader(sigma_noise=self.sigma_w)
        self.traders_order = np.random.choice(["i", "u"], size=self.tmax, p=[self.alpha, 1-self.alpha])

        
        for t in tqdm([t for t in range(self.tmax)]):
            
            ## if jump, reset the prior distribution on V
            if self.jumps[t]==1:
                self.v_distrib.reset(centered_at=self.exp_value[-1])
            
            ## MM sets bid and ask price
            curr_ask = fixed_point(Pa_fp, self.true_value[t], args=(self.alpha, self.eta, self.sigma_w, self.v_distrib.vec_v, self.v_distrib.prior_v), xtol=1e-2, maxiter=500, method='del2').item()
            curr_ask += -self.gamma*self.inventory[-1]
            curr_ask += self.extend_spread ## extend spread
            self.asks.append(curr_ask)

            curr_bid = fixed_point(Pb_fp, self.true_value[t], args=(self.alpha, self.eta, self.sigma_w, self.v_distrib.vec_v, self.v_distrib.prior_v), xtol=1e-2, maxiter=500, method='del2').item()
            curr_bid += -self.gamma*self.inventory[-1]
            curr_bid += -self.extend_spread ## extend spread
            self.bids.append(curr_bid)


            ## priors or buying, selling, no order
            Pbuy = P_buy(Pa=self.asks[-1], alpha=self.alpha, eta=self.eta, sigma_w=self.sigma_w, vec_v=self.v_distrib.vec_v, v_prior=self.v_distrib.prior_v)
            assert Pbuy>0-self.eps and Pbuy<1+self.eps, "Pbuy not between 0 and 1"
            
            Psell = P_sell(Pb=self.bids[-1], alpha=self.alpha, eta = self.eta, sigma_w=self.sigma_w, vec_v=self.v_distrib.vec_v, v_prior=self.v_distrib.prior_v)
            assert Psell>0-self.eps and Psell<1+self.eps, "Psell not between 0 and 1"
            
            Pnoorder = P_no_order(Pb=self.bids[-1], Pa=self.asks[-1], alpha=self.alpha, eta=self.eta, sigma_w=self.sigma_w, vec_v=self.v_distrib.vec_v, v_prior=self.v_distrib.prior_v)
            assert Pnoorder>0-self.eps and Pnoorder<1+self.eps, "P_noorder not between 0 and 1"

            assert Psell+Pbuy+Pnoorder>0-self.eps and Pbuy +Psell+Pnoorder<1+self.eps, "sum of order priors not between 0 and 1"

            ## compute expected value
            self.exp_value.append(compute_exp_true_value(Pb=self.bids[-1], 
                                                    Pa=self.asks[-1],
                                                    psell=Psell, 
                                                    pbuy=Pbuy,
                                                    vec_v=self.v_distrib.vec_v,
                                                    v_prior=self.v_distrib.prior_v,
                                                    alpha=self.alpha,
                                                    eta=self.eta,
                                                    sigma_w=self.sigma_w))

            ## tarders trade
            if self.traders_order[t] == "i":
                trade = self.i_trader.trade(bid=self.bids[-1], ask=self.asks[-1], true_value=self.true_value[t])
                self.u_trader.update_pnl(0, 45, 45) ## not useful (abitrary)
            else:
                trade = self.u_trader.trade()
                self.i_trader.update_pnl(0, 45, 45) ## not useful (abitrary)

            ## Update MM pnl
            self.update_pnl_and_inventory(trader_direction=trade, bid=self.bids[-1], ask=self.asks[-1], true_value=self.true_value[t])


            ## update MM proba distribution with trade info
            self.v_distrib.compute_posterior(trade, 
                                            Pbuy=Pbuy, 
                                            Psell=Psell, 
                                            Pno=Pnoorder, 
                                            Pa=self.asks[-1], 
                                            Pb=self.bids[-1],
                                            alpha=self.alpha, 
                                            eta=self.eta, 
                                            sigma_w=self.sigma_w,
                                            update_prior=True)

            assert np.abs(sum(self.v_distrib.prior_v)-1) < self.eps, "posterior prob is not normalized"

        self.run = True # simulation finised
        print("Simulation finsihed")