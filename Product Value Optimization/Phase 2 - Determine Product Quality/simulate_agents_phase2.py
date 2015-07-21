import sys
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from agents import RatioAgent, NaiveBayesAgent
#=======================
from agent_skatkar import *
#=======================

def simulate_agents(agents, value, X, y, price_trials = 10):
    
    agent_wealths = {}
    
    for agent in agents:
        agent_wealths[agent] = 0
    
    num_products = X.shape[0]
    
    for p in range(num_products):        
        
        # working or not?
        working = (y[p] == 1)
        
        for agent in agents:
            prob = agent.predict_prob_of_good(X[p])
            # try a range of prices            
            for pt in range(price_trials):                            
                price = ((2*pt+1)*value)/(2*price_trials)                                
                if agent.will_buy(value, price, prob):
                    agent_wealths[agent] -= price
                    if working:
                        agent_wealths[agent] += value
    return agent_wealths

if __name__ == '__main__':
    data_path = "./"
    data_group = "dataset12"
    # if you like, you can read the data_path and data_group using sys.argv
    
    X_train_file = data_path + data_group +  "_X_train.csv"
    y_train_file = data_path + data_group + "_y_train.csv"   
    X_val_file = data_path + data_group + "_X_val.csv"
    y_val_file = data_path + data_group + "_y_val.csv"
    X_test_file = data_path + data_group + "_X_test.csv"
    
    X_train = np.loadtxt(X_train_file, dtype=float, delimiter=',')
    y_train = np.loadtxt(y_train_file, dtype=int, delimiter=',')
    X_val = np.loadtxt(X_val_file, dtype=float, delimiter=',')
    y_val = np.loadtxt(y_val_file, dtype=int, delimiter=',')
    X_test = np.loadtxt(X_test_file, dtype=float, delimiter=',')
    
    agents = []
    
    
    agents.append(RatioAgent("ratio_0.75", 0.75))
    agents.append(RatioAgent("ratio_0.50", 0.5))
    agents.append(RatioAgent("ratio_0.25", 0.25))
    agents.append(NaiveBayesAgent("nb"))
    agents.append(Agent_skatkar("My_classifier"))

    
    # Train the agents
    for agent in agents:
        agent.fit_a_classifier(X_train, y_train, X_val, y_val)
    
    # Simulate the agents
    value = 1000
    agent_wealths = simulate_agents(agents, value, X_val, y_val)
    
    print "-" * 50
    print "SIMULATION RESULTS ON THE VALIDATION DATA"
    print "-" * 50
    
    for agent in agents:
        print "{}:\t\t${:,.2f}".format(agent, agent_wealths[agent])
    
    print
    print "-" * 50
    print "PREDICTED PROBABILITIES ON THE TEST DATA"
    print "-" * 50
    
    student_agent = agents[4]
    
    prob_list=[]
    for i in range(X_test.shape[0]):
        prob = student_agent.predict_prob_of_good(X_test[i])
        prob_list.append(prob)
    print prob_list


   

    