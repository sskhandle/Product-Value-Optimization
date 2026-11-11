import numpy as np

from agents import RatioAgent, NaiveBayesAgent
from agent_skatkar import Agent_skatkar


def simulate_agents(agents, value, X, y, price_trials=10):
    agent_wealths = dict.fromkeys(agents, 0)

    for x, label in zip(X, y):
        for agent in agents:
            prob = agent.predict_prob_of_good(x)
            for pt in range(price_trials):
                price = (2 * pt + 1) * value / (2 * price_trials)
                if agent.will_buy(value, price, prob):
                    agent_wealths[agent] -= price
                    if label == 1:
                        agent_wealths[agent] += value
    return agent_wealths


if __name__ == '__main__':
    group = "./dataset12"
    X_train = np.loadtxt(group + "_X_train.csv", dtype=float, delimiter=',')
    y_train = np.loadtxt(group + "_y_train.csv", dtype=int, delimiter=',')
    X_val = np.loadtxt(group + "_X_val.csv", dtype=float, delimiter=',')
    y_val = np.loadtxt(group + "_y_val.csv", dtype=int, delimiter=',')
    X_test = np.loadtxt(group + "_X_test.csv", dtype=float, delimiter=',')

    my_agent = Agent_skatkar("My_classifier")
    agents = [
        RatioAgent("ratio_0.75", 0.75),
        RatioAgent("ratio_0.50", 0.5),
        RatioAgent("ratio_0.25", 0.25),
        NaiveBayesAgent("nb"),
        my_agent,
    ]

    for agent in agents:
        agent.fit_a_classifier(X_train, y_train, X_val, y_val)

    wealths = simulate_agents(agents, 1000, X_val, y_val)

    print("\nResults on validation data")
    for agent in agents:
        print(f"{agent}:\t\t${wealths[agent]:,.2f}")

    print("\nPredicted probabilities on test data")
    print([my_agent.predict_prob_of_good(x) for x in X_test])

