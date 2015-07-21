import numpy as np

from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from agents import Agent
from simulate_agents_phase2 import *

class Agent_skatkar(Agent):
    
    data_path = "./"
    data_group = "dataset12"
    
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
        
    
    def find_best(X_train, y_train, X_validation, y_validation):
        classifiers = [
            LogisticRegression(),
            KNeighborsClassifier(3),
            KNeighborsClassifier(n_neighbors=7, weights="uniform"),
            KNeighborsClassifier(n_neighbors=10, weights="uniform"),
            KNeighborsClassifier(n_neighbors=3, weights="uniform"),
            KNeighborsClassifier(n_neighbors=7, weights="distance"),
            KNeighborsClassifier(n_neighbors=10, weights="distance"),
            KNeighborsClassifier(n_neighbors=3, weights="uniform"),
            SVC(kernel="linear", C=0.025, probability = True),
            SVC(kernel = "rbf", C=10, gamma=0.01, probability=True),
            SVC(kernel = "rbf", C=1, gamma=0.01, probability=True),
            SVC(gamma=2, C=1, probability = True),
            DecisionTreeClassifier(max_depth=5),
            DecisionTreeClassifier(max_depth=1, criterion='entropy'),
            DecisionTreeClassifier(max_depth=5, criterion='entropy'),
            DecisionTreeClassifier(max_depth=10, criterion='entropy'),
            DecisionTreeClassifier(max_depth=5, criterion='entropy'),
            DecisionTreeClassifier(max_depth=10, criterion='gini'),
            DecisionTreeClassifier(max_depth=5, criterion='gini'),
            DecisionTreeClassifier(max_depth=1, criterion='gini'),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            RandomForestClassifier(max_depth=5, n_estimators=30, max_features=5, criterion = 'gini'),
            RandomForestClassifier(max_depth=5, n_estimators=20, max_features=10, criterion = 'entropy'),
            RandomForestClassifier(max_depth=5, n_estimators=30, max_features=10, criterion = 'gini'),
            RandomForestClassifier(max_depth=5, n_estimators=20, max_features=15, criterion = 'entropy'),
            RandomForestClassifier(max_depth=5, n_estimators=20, max_features=10, criterion = 'gini'),
            RandomForestClassifier(max_depth=5, n_estimators=30, max_features=15, criterion = 'entropy'),
            AdaBoostClassifier(),
            GaussianNB(),
            LDA(),
            QDA(),
            QDA(reg_param = 0.001),
            QDA(reg_param = 0.1),
            QDA(reg_param = 0.01),
            SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
            kernel='rbf', max_iter=-1, probability=True, random_state=None,
            shrinking=True, tol=0.001, verbose=False)]
            
        clf_dict = {}
        y_pred_list = []
        
        for clf in classifiers:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_validation)
            y_pred_list.append(y_pred)
            acc = metrics.accuracy_score(y_validation, y_pred)
            # avg_prec = metrics.average_precision_score(y_validation, y_pred)
            # prec = metrics.precision_score(y_validation, y_pred)
            # class_rep = metrics.classification_report(y_validation, y_pred, target_names=['background', 'foreground'])
            # f1 = metrics.f1_score(y_validation, y_pred)
            
            clf_dict[clf] = acc
            
        global best_one
        best_one = max(clf_dict, key=clf_dict.get)        
        print("{" + "\n".join("{}: {}".format(k, v) for k, v in clf_dict.items()) + "}")
        print("\n\n********THE BEST CLASSIFIER IS********\n")
        print(best_one)
                
    find_best(X_train, y_train, X_val, y_val)    

    
    def fit_a_classifier(self, X_train, y_train, X_validation, y_validation):
        self.classifier = best_one
        self.classifier.fit(X_train, y_train)
        global y_pred
        y_pred = self.classifier.predict(X_validation)
        print y_pred
        
        print("\n******CONFUSION MATRIX*****\n")
        cm = confusion_matrix(y_validation, y_pred)
        print(cm)   

        thefile = open('test_pred.txt', 'w')
        thefile.write("\n".join(str(i) for i in y_pred))
        thefile.close()        