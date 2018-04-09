import csv

names = []
data = []

with open('data/spambase/spambase.names', 'rt') as f:
    spamreader = csv.reader(f, delimiter=',', quotechar='~')
    for row in spamreader:
        names.append(row[0].strip())

with open('data/spambase/spambase.data', 'rt') as f:
    spamreader = csv.reader(f, delimiter=',', quotechar='~')
    for row in spamreader:
        data.append({name:row[k] for (k,name) in enumerate(names)})

print(data)