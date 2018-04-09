import csv

names = []
data = []

with open('data/spambase/spambase.formatted_names', 'rt') as f:
    names = [name.strip() for name in f.readlines()]

with open('data/spambase/spambase.data', 'rt') as f:
    data_reader = csv.reader(f, delimiter=',', quotechar='~')
    data = list(map(lambda input: {name : float(input[k]) for (k , name) in enumerate(names)}, data_reader))

spam = list(filter(lambda x: x["is_spam"] == 1, data))
regular = list(filter(lambda x: x["is_spam"] == 0, data))


def get_perf(n, d):
    return sum([input[n] for input in d]) / len(d) * 100

def sort_perf(d):
    return sorted({ n:get_perf(n,d) for n in names }, key=lambda name: get_perf(name, d))

print(get_perf("word_freq_george", data))

print(sort_perf(data))