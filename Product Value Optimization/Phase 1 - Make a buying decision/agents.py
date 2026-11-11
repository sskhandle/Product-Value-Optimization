class Agent:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "Agent_" + self.name

    def will_buy(self, value, price, prob):
        raise NotImplementedError


class HalfProbAgent(Agent):
    """Buys if prob > 0.5, ignoring value and price."""

    def will_buy(self, value, price, prob):
        return prob > 0.5


class RatioAgent(Agent):
    """Buys if price/value is under a threshold."""

    def __init__(self, name, max_p_v_ratio):
        super().__init__(name)
        self.max_p_v_ratio = max_p_v_ratio

    def will_buy(self, value, price, prob):
        return price / value <= self.max_p_v_ratio


class BuyAllAgent(Agent):
    def will_buy(self, value, price, prob):
        return True

