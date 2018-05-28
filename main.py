from os import chdir
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

chdir('./spambase/')

with open('spambase.formatted_names', 'rt') as f:
    names = [name.strip() for name in f.readlines()]

data = pd.read_csv('spambase.data', names=names)

# data.head()
# data.shape
# data.info()
# data.describe()
# data[data.columns[57]].value_counts(1)

target = data[data.columns[-1]]
X = data.drop(data.columns[-1], axis=1)

# X.shape

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

# y_train.value_counts(1)
# y_test.value_counts(1)

clf_LR = LogisticRegression()
clf_LR.fit(X_train, y_train)
output_LR = clf_LR.predict(X_test)

accuracy_LR = accuracy_score(y_test, output_LR)
precision_LR = precision_score(y_test, output_LR)
recall_LR = recall_score(y_test, output_LR)
f1_score = f1_score(y_test, output_LR)
auc = roc_auc_score(y_test, output_LR)

print('accuracy ', accuracy_LR, 'precision ', precision_LR, 'recall ', recall_LR, 'f1_score ', f1_score, 'auc ', auc)