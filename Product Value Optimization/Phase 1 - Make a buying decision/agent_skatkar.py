from agents import Agent


class Agent_skatkar(Agent):
    def will_buy(self, value, price, prob):
        """Buy when the chance of it working beats price/value."""
        return prob > price / value

