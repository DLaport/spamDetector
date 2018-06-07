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

    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)


init()
algos.run_algos()
graphs.wordcloud()

#
# categories = spam.columns
# N = categories.size
#
# # y-axis in bold
# rc('font', weight='bold')
#
# for name in names:
#     column = spam[name]
#     print(column.idxmax() == 0)
#
# # library & dataset
# import seaborn as sns
#
# df = sns.load_dataset('iris')
# print(df)
# sns.boxplot(data=spam.ix[:, 54:57], showfliers=False)
# plt.show()


# barWidth = 0.25
#
# # set height of bar
# bars1 = [12, 30, 1, 8, 22]
# bars2 = [28, 6, 16, 5, 10]
#
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
#
# # Make the plot
# plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='var1')
# plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='var2')
#
# # Add xticks on the middle of the group bars
# plt.xlabel('group', fontweight='bold')
# plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])
#
# # Create legend & Show graphic
# plt.legend(["spam", "non_spam"])
# plt.show()
#
