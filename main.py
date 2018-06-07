import os
os.chdir('./spambase/')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('spambase.data',header = None)

data.head()

#data.shape

#data.info()

data.describe()

# DATA CLEANING
names = pd.read_csv('spambase.formatted_names', header = None)[0]

spam = data.loc[data[57] == 1]
spam_feature_means = spam.mean(axis=0)

regular = data.loc[data[57] == 0]
regular_feature_means = regular.mean(axis=0)

means_ratios = [
    max(spam_feature_means[i], regular_feature_means[i]) / min(spam_feature_means[i], regular_feature_means[i])
    for i in range(0,56)]

del names[56]
del names[57]
plt.stem(names, means_ratios)
plt.ylim(0, 3)
#plt.show()

names_ratios = [(names[i], means_ratios[i]) for i in range(0,56)]
sorted_ratios = sorted(names_ratios, key=lambda nr: nr[1])

print(sorted_ratios)

for i in range(56):
    if means_ratios[i] < 20:
        data.drop(i, 1)

data[data.columns[57]].value_counts(1)

target = data[data.columns[-1]]
X = data.drop(data.columns[-1],axis = 1)

X.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,target, test_size=0.2, random_state=42)

y_train.value_counts(1)

y_test.value_counts(1)

from sklearn.linear_model import LogisticRegression

clf_LR = LogisticRegression()

clf_LR.fit(X_train,y_train)

output_LR = clf_LR.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score,roc_auc_score

accuracy_LR = accuracy_score(y_test,output_LR)
precision_LR = precision_score(y_test,output_LR)
recall_LR = recall_score(y_test,output_LR)
f1_score = f1_score(y_test,output_LR)
auc = roc_auc_score(y_test,output_LR)
print('accuracy ',accuracy_LR,'precision ',precision_LR,'recall ', recall_LR, 'f1_score ', f1_score,'auc ',auc)