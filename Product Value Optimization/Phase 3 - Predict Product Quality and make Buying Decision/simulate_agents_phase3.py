import numpy as np

from product import Product
from agent_skatkar import Agent_skatkar


if __name__ == '__main__':
    group = "./dataset1"
    X = np.loadtxt(group + "_X.csv", dtype=float, delimiter=',')
    y = np.loadtxt(group + "_y.csv", dtype=int)  # 1 = good, 0 = bad
    prices = np.loadtxt(group + "_p.csv", dtype=float)

    value = 1000.
    num_products = len(X)
    products = [Product(x, value, p) for x, p in zip(X, prices)]

    agent = Agent_skatkar("Agent_skatkar")
    wealth = 0
    num_good = 0
    rs = np.random.RandomState(42)

    # one good and one bad product are handed over for free
    for label in (1, 0):
        i = rs.choice(np.nonzero(y == label)[0])
        products[i].price = 0
        agent.add_to_my_products(products[i], label)
        if label == 1:
            wealth += value
            num_good += 1
        del products[i]
        y = np.delete(y, i)

    for _ in range(num_products // 2):
        i = agent.choose_one_product(products)
        agent.add_to_my_products(products[i], y[i])
        wealth -= products[i].price
        if y[i] == 1:
            wealth += products[i].value
            num_good += 1
        del products[i]
        y = np.delete(y, i)

    print(f"{agent}'s final wealth:\t${wealth:,.2f}")
    print(f"{agent} has {num_good} good products.")

