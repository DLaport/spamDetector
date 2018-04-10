from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import urllib
import csv
url = "../data/spambase/spambase.data"


namesCat = []

#data = pd.read_csv(url, names=namesCat)

data = pd.read_csv("../data/spambase/spambase.data", header=None)
data.columns = ["V"+str(i) for i in range(1, len(data.columns)+1)]  # rename column names to be similar to R naming convention
data.V1 = data.V1.astype(str)
X = data.loc[:, "V2":]  # independent variables data
y = data.V1  # dependednt variable data
pd.tools.plotting.scatter_matrix(data.loc[:, "V58":], diagonal="kde")
plt.tight_layout()
plt.show()
print(data)




