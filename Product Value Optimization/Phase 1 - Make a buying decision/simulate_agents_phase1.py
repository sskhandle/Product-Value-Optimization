import numpy as np

from agents import HalfProbAgent, RatioAgent, BuyAllAgent
from agent_skatkar import Agent_skatkar


def simulate_agents(agents, value, num_products, alpha, beta, seed=None):
    agent_wealths = dict.fromkeys(agents, 0)

    if seed is not None:
        np.random.seed(seed)

    for _ in range(num_products):
        price = np.random.rand() * value
        prob = np.random.beta(alpha, beta)
        working = prob > np.random.rand()

        for agent in agents:
            if agent.will_buy(value, price, prob):
                agent_wealths[agent] -= price
                if working:
                    agent_wealths[agent] += value

    return agent_wealths


if __name__ == '__main__':
    value = 1000.
    num_products = 1000

    agents = [
        HalfProbAgent("hp"),
        RatioAgent("ratio_0.75", 0.75),
        RatioAgent("ratio_0.50", 0.5),
        RatioAgent("ratio_0.25", 0.25),
        BuyAllAgent("buy_all"),
        Agent_skatkar("agent_skatkar"),
    ]

    # good:bad ratio is 1:1, 1:2 and 2:1
    markets = [("FAIR MARKET", 1, 1), ("JUNK YARD", 1, 2), ("FANCY MARKET", 2, 1)]

    for title, alpha, beta in markets:
        wealths = simulate_agents(agents, value, num_products, alpha, beta)
        print(f"\n{title}")
        for agent in agents:
            print(f"{agent}:\t\t${wealths[agent]:,.2f}")

