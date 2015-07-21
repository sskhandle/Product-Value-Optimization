from agents import Agent

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import numpy as np

class Agent_skatkar(Agent):

    def __init__(self, name, seed=0):
        super(Agent_skatkar, self).__init__(name)
        
        self.clf = BernoulliNB()
        # self.clf = LogisticRegression()
        # self.clf = GaussianNB()
        # self.clf = KNeighborsClassifier()
        # self.clf = SVC()
        # self.clf = DecisionTreeClassifier()
        # self.clf = RandomForestClassifier()
        # self.clf = AdaBoostClassifier()
            
    def choose_one_product(self, products):
                
        XX = [i.features for i in self.my_products]
            
        yy = [i for i in self.product_labels]
                
        self.clf.fit(XX, yy)

        v = 0
        g_val = 0
        for i in range(len(products)):
            feat = self.clf.predict_proba(products[i].features)[0][1]
            val = products[i].value
            cost = products[i].price
            prob = feat
            temp = prob*(val - cost)
            if(temp > v):
                v = temp
                g_val = i        
        return g_val
    
                
        