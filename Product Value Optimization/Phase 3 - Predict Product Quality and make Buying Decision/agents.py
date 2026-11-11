import numpy as np


class Agent:
    def __init__(self, name):
        self.name = name
        self.my_products = []
        self.product_labels = []

    def __repr__(self):
        return "Agent_" + self.name

    def choose_one_product(self, products):
        raise NotImplementedError

    def add_to_my_products(self, product, label):
        self.my_products.append(product)
        self.product_labels.append(label)


class CheapAgent(Agent):
    """Picks the cheapest product."""

    def choose_one_product(self, products):
        return min(range(len(products)), key=lambda i: products[i].price)


class RandomAgent(Agent):
    def __init__(self, name, seed=0):
        super().__init__(name)
        self.rg = np.random.RandomState(seed)

    def choose_one_product(self, products):
        return self.rg.choice(len(products))

