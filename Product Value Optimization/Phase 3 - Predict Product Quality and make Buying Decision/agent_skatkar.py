from sklearn.naive_bayes import BernoulliNB

from agents import Agent


class Agent_skatkar(Agent):
    def __init__(self, name):
        super().__init__(name)
        self.clf = BernoulliNB()

    def choose_one_product(self, products):
        """Refit on owned products, then pick the best prob * (value - price)."""
        self.clf.fit([p.features for p in self.my_products], self.product_labels)

        best, best_gain = 0, 0
        for i, p in enumerate(products):
            prob = self.clf.predict_proba([p.features])[0][1]
            gain = prob * (p.value - p.price)
            if gain > best_gain:
                best, best_gain = i, gain
        return best

