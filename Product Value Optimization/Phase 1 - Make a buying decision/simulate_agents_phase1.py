'''
Simulate agents - Phase 1
'''

import numpy as np
from agents import HalfProbAgent, RatioAgent, BuyAllAgent
from agent_skatkar import Agent_skatkar
from pp1_r2 import Agent_skatkar1

def simulate_agents(agents, value, num_products, alpha, beta, seed=None):
    
    agent_wealths = {}
    
    for agent in agents:
        agent_wealths[agent] = 0
    
    if seed:
        np.random.seed(seed)
    
    for _ in range(num_products):
        # price is always lower than or equal to the value
        price = np.random.rand()*value
        
        # prob of working
        prob = np.random.beta(alpha, beta)
        
        # working or not?
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
    
    agents = []
    
    agents.append(HalfProbAgent("hp"))
    
    agents.append(RatioAgent("ratio_0.75", 0.75))
    agents.append(RatioAgent("ratio_0.50", 0.5))
    agents.append(RatioAgent("ratio_0.25", 0.25))
    agents.append(BuyAllAgent("buy_all"))
    # add our own agent   
    #agents.append(Agent_<hawk_username>("agent_<hawk_username>"))
    agents.append(Agent_skatkar("agent_skatkar"))
    agents.append(Agent_skatkar1("agent_skatkar"))
    
    # Fair market; the ratio of Good to Bad products is 1:1
    agent_wealths = simulate_agents(agents, value, num_products, 1, 1)
    
    print '-' * 50
    print 'FAIR MARKET'
    print '-' * 50
    for agent in agents:
        print "{}:\t\t${:,.2f}".format(agent, agent_wealths[agent])
    
    # Junk market; the ratio of Good to Bad products is 1:2
    agent_wealths = simulate_agents(agents, value, num_products, 1, 2)
    
    print
    print '-' * 50
    print 'JUNK YARD'
    print '-' * 50
    for agent in agents:
        print "{}:\t\t${:,.2f}".format(agent, agent_wealths[agent])
    
    # Fancy market; the ratio of Good to Bad products is 2:1
    agent_wealths = simulate_agents(agents, value, num_products, 2, 1)
    
    print
    print '-' * 50
    print 'FANCY MARKET'
    print '-' * 50
    for agent in agents:
        print "{}:\t\t${:,.2f}".format(agent, agent_wealths[agent])