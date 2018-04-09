import csv

names = []
data = []

with open('data/spambase/formated_spambase.names', 'rt') as f:
    spamreader = csv.reader(f, delimiter=',', quotechar='~')
    for row in spamreader:
        names.append(row[0].strip())

with open('data/spambase/spambase.data', 'rt') as f:
    spamreader = csv.reader(f, delimiter=',', quotechar='~')
    for row in spamreader:
        data.append({name:row[k] for (k,name) in enumerate(names)})

def get_perf(name):
    spam_perf = sum([float(input[name]) for input in data if input["is_spam"] == "1"]) / len(data)
    regular_perf = sum([float(input[name]) for input in data if input["is_spam"] == "0"]) / len(data)
    return (spam_perf, regular_perf)

print(get_perf("word_freq_make"))
