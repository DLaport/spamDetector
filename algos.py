import main

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score,roc_auc_score


def is_ambiguous(element):
    return 0.4 < element < 0.6

def is_obvious(element):
    return element < 0.1 or element > 0.9


def run_algos():

    for (name, algo) in [("Logistic Regression", LogisticRegression()),
                         ("Random Forest Classifier", RandomForestClassifier()),
                         ("XGB Classifier", XGBClassifier())]:
        algo.fit(main.X_train, main.y_train)
        output = algo.predict(main.X_test)
        probas = algo.predict_proba(main.X_test)
        ambiguous = [index for index, el in enumerate(probas) if is_ambiguous(el[0])]
        obvious_LR = [index for index, el in enumerate(probas) if is_obvious(el[0])]
        accuracy_RF = accuracy_score(main.y_test, output)
        precision_RF = precision_score(main.y_test, output)
        recall_RF = recall_score(main.y_test, output)
        f1_score_RF = f1_score(main.y_test, output)
        auc_RF = roc_auc_score(main.y_test, output)
        print(name + ' accuracy ', accuracy_RF, 'precision ', precision_RF, 'recall ', recall_RF, 'f1_score ', f1_score_RF,
              'auc ', auc_RF)


# agreed_upon_ambiguity = set(ambiguous_XG) & set(ambiguous_RF) & set(ambiguous_LR)
# print(list(agreed_upon_ambiguity))  # algorithms don't agree on which entries are harder to predict
# agreed_upon_obviousness = set(obvious_XG) & set(obvious_RF) & set(obvious_LR)
# print(agreed_upon_obviousness)
# print(len(agreed_upon_obviousness))

