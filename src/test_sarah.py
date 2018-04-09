
import csv


data_file = open('../data/spambase/spambase.data', 'rt')
name_file = open('../data/spambase/spambase.formatted_names', 'rt')

data_reader = csv.reader(data_file, delimiter=',', quotechar='|')
name_reader = [x.strip() for x in name_file.readlines()]


def to_dictionary(line):
    dictionary = {}
    for i in range(0, len(line)):
        dictionary[name_reader[i]] = line[i]
    return dictionary


data = list(map(lambda x: to_dictionary(x), data_reader))

#  for entry in data[:10]:
#    print(entry)

spam = list(filter(lambda x: x["is_spam"] == '1', data))
not_spam = list(filter(lambda x: x["is_spam"] == 0, data))

print(len(spam)) # 1813
