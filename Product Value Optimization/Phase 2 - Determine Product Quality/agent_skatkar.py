import numpy as np
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from agents import Agent


class Agent_skatkar(Agent):
    def _candidates(self):
        knn = [KNeighborsClassifier(n_neighbors=n, weights=w)
               for n in (3, 7, 10) for w in ("uniform", "distance")]
        trees = [DecisionTreeClassifier(max_depth=d, criterion=c)
                 for d in (1, 5, 10) for c in ("gini", "entropy")]
        forests = [RandomForestClassifier(max_depth=5, n_estimators=n, max_features=f, criterion=c)
                   for n, f, c in [(10, 1, "gini"), (30, 5, "gini"), (20, 10, "entropy"),
                                   (30, 10, "gini"), (20, 15, "entropy"), (20, 10, "gini"),
                                   (30, 15, "entropy")]]
        svcs = [SVC(kernel="linear", C=0.025, probability=True),
                SVC(kernel="rbf", C=10, gamma=0.01, probability=True),
                SVC(kernel="rbf", C=1, gamma=0.01, probability=True),
                SVC(gamma=2, C=1, probability=True),
                SVC(C=10, probability=True)]
        qdas = [QuadraticDiscriminantAnalysis(reg_param=r) for r in (0, 0.001, 0.01, 0.1)]
        return (knn + trees + forests + svcs + qdas +
                [LogisticRegression(), AdaBoostClassifier(), GaussianNB(),
                 LinearDiscriminantAnalysis()])

    def fit_a_classifier(self, X_train, y_train, X_validation, y_validation):
        """Fit every candidate and keep the one with the best validation accuracy."""
        best_acc = -1
        for clf in self._candidates():
            clf.fit(X_train, y_train)
            acc = metrics.accuracy_score(y_validation, clf.predict(X_validation))
            if acc > best_acc:
                best_acc, self.classifier = acc, clf

        print("Best classifier:", self.classifier, f"(val accuracy {best_acc:.3f})")
        y_pred = self.classifier.predict(X_validation)
        print(metrics.confusion_matrix(y_validation, y_pred))
        np.savetxt("test_pred.txt", y_pred, fmt="%d")

