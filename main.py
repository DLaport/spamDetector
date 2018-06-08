from pandas import read_csv
from sklearn.model_selection import train_test_split
from os import chdir
import re
import algos
import graphs


def init():
    global data, target, X, names
    global X_train, X_test, y_train, y_test
    global spam, non_spam

    chdir('./spambase/')

    with open('spambase.formatted_names', 'rt') as f:
        names = [name.strip() for name in f.readlines()]

    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    names = [regex.sub("_", col) if any(x in str(col) for x in {'[', ']', '<'}) else col for col in names]
    # xgBoost has issues with column labels containing certain characters..

    data = read_csv('spambase.data', names=names)

    spam = data[data["is_spam"] == 1]
    non_spam = data[data["is_spam"] == 0]

    target = data[data.columns[-1]]
    X = data.drop(data.columns[-1], axis=1)

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    X = SelectKBest(chi2, k=42).fit_transform(X, target)

    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)


init()
matrixes = algos.run()
graphs.matrixes(matrixes)
#graphs.wordcloud()