'''
The agent base class as well as the baseline agents.
'''

def abstract():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')


class Agent(object):
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return "Agent_" + self.name
    
    def will_buy(self, value, price, prob):
        abstract()

class HalfProbAgent(Agent):
    """Buys if the prob > 0.5 no matter what the value or price is"""
    
    def will_buy(self, value, price, prob):
        return (prob > 0.5)

class RatioAgent(Agent):
    """Buys if the ratio of the price to value is below a specified threshold"""
    
    def __init__(self, name, max_p_v_ratio):
        super(RatioAgent, self).__init__(name)
        self.max_p_v_ratio = max_p_v_ratio
    
    def will_buy(self, value, price, prob):
        return (price/value <= self.max_p_v_ratio)

class BuyAllAgent(Agent):
    """Simply buys all products"""
    
    def will_buy(self, value, price, prob):
        return True   

