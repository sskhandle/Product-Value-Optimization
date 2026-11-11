from sklearn.naive_bayes import BernoulliNB


class Agent:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "Agent_" + self.name

    def will_buy(self, value, price, prob):
        return value * prob > price

    def fit_a_classifier(self, X_train, y_train, X_validation, y_validation):
        """Must set self.classifier to a fitted sklearn classifier with predict_proba."""
        raise NotImplementedError

    def predict_prob_of_good(self, x):
        return self.classifier.predict_proba([x])[0][1]


class RatioAgent(Agent):
    """Buys if price/value is under a threshold."""

    def __init__(self, name, max_p_v_ratio):
        super().__init__(name)
        self.max_p_v_ratio = max_p_v_ratio

    def fit_a_classifier(self, X_train, y_train, X_validation, y_validation):
        pass

    def predict_prob_of_good(self, x):
        return 0

    def will_buy(self, value, price, prob):
        return price / value <= self.max_p_v_ratio


class NaiveBayesAgent(Agent):
    def fit_a_classifier(self, X_train, y_train, X_validation, y_validation):
        self.classifier = BernoulliNB().fit(X_train, y_train)

