import numpy as np
        np.savetxt("test_pred.txt", y_pred, fmt="%d")
        print(metrics.confusion_matrix(y_validation, y_pred))
        y_pred = self.classifier.predict(X_validation)
        print("Best classifier:", self.classifier, f"(val accuracy {best_acc:.3f})")

                best_acc, self.classifier = acc, clf
            if acc > best_acc:
            acc = metrics.accuracy_score(y_validation, clf.predict(X_validation))
            clf.fit(X_train, y_train)
        for clf in self._candidates():
        best_acc = -1
        """Fit every candidate and keep the one with the best validation accuracy."""
    def fit_a_classifier(self, X_train, y_train, X_validation, y_validation):

                 LinearDiscriminantAnalysis()])
                [LogisticRegression(), AdaBoostClassifier(), GaussianNB(),
        return (knn + trees + forests + svcs + qdas +
        qdas = [QuadraticDiscriminantAnalysis(reg_param=r) for r in (0, 0.001, 0.01, 0.1)]
                SVC(C=10, probability=True)]
                SVC(gamma=2, C=1, probability=True),
                SVC(kernel="rbf", C=1, gamma=0.01, probability=True),
                SVC(kernel="rbf", C=10, gamma=0.01, probability=True),
        svcs = [SVC(kernel="linear", C=0.025, probability=True),
                                   (30, 15, "entropy")]]
                                   (30, 10, "gini"), (20, 15, "entropy"), (20, 10, "gini"),
                   for n, f, c in [(10, 1, "gini"), (30, 5, "gini"), (20, 10, "entropy"),
        forests = [RandomForestClassifier(max_depth=5, n_estimators=n, max_features=f, criterion=c)
                 for d in (1, 5, 10) for c in ("gini", "entropy")]
        trees = [DecisionTreeClassifier(max_depth=d, criterion=c)
               for n in (3, 7, 10) for w in ("uniform", "distance")]
        knn = [KNeighborsClassifier(n_neighbors=n, weights=w)
    def _candidates(self):
class Agent_skatkar(Agent):


from agents import Agent

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import metrics

