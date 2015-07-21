'''
The agent base class as well as the baseline agents.
'''

import numpy as np

def abstract():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')


class Agent(object):
    def __init__(self, name):
        self.name = name
        self.my_products = []
        self.product_labels = []
    
    def __repr__(self):
        return "Agent_" + self.name
    
    def choose_one_product(self, products):
        abstract()
        
    def add_to_my_products(self, product, label):
        self.my_products.append(product)
        self.product_labels.append(label)

class CheapAgent(Agent):
    """Chooses the cheapest product"""
    
    def choose_one_product(self, products):
        cheapest_price = np.Inf
        chosen_product = None
        for i in range(len(products)):
            if products[i].price < cheapest_price:
                cheapest_price = products[i].price
                chosen_product = i
        return chosen_product

class RandomAgent(Agent):
    """Chooses a random product"""
    def __init__(self, name, seed=0):
        super(RandomAgent, self).__init__(name)
        self.rg = np.random.RandomState(seed)
    
    def choose_one_product(self, products):
        return self.rg.choice(len(products))