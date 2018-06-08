import main

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

chosen_algos = [
    ("Logistic Regression", LogisticRegression()),
    ("Random Forest Classifier", RandomForestClassifier()),
    ("XGB Classifier", XGBClassifier())
]


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    ("Nearest Neighbors", KNeighborsClassifier(3)),
    ("Linear SVM", SVC(kernel="linear", C=0.025)),
    ("RBF SVM", SVC(gamma=2, C=1)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
    ("Neural Net", MLPClassifier(alpha=1)),
    ("AdaBoost", AdaBoostClassifier()),
    ("Naive Bayes", GaussianNB()),
    ("QDA", QuadraticDiscriminantAnalysis()),
    ("Logistic Regression", LogisticRegression()),
    ("Random Forest Classifier", RandomForestClassifier()),
    ("XGB Classifier", XGBClassifier()),
]


def run():
    return run_algos(classifiers)


def run_algos(algo_list):
    confusion_matrixes = {}
    for (name, algo) in algo_list:

        algo.fit(main.X_train, main.y_train)
        output = algo.predict(main.X_test)
        # probas = algo.predict_proba(main.X_test)
        accuracy = round(accuracy_score(main.y_test, output), 4)*100
        precision = round(precision_score(main.y_test, output), 4)*100
        recall = round(recall_score(main.y_test, output), 4)*100
        f1_score_ = round(f1_score(main.y_test, output), 4)*100
        auc = round(roc_auc_score(main.y_test, output), 4)*100

        print(name, ':  Accuracy - ', accuracy, ', Precision - ', precision, ', Recall - ', recall,
              ', F1_score - ', f1_score_, ' AUC - ', auc)

        confusion_matrixes[name] = (confusion_matrix(main.y_test, output))

    return confusion_matrixes

