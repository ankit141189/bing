from collections import defaultdict
import pandas as pd

file = open('BingHackathonTrainingData.txt')

table = defaultdict(lambda : defaultdict(int))

for line in file.readlines():
	fields = line.rstrip().split('\t')
	table[fields[2]][fields[1]] += 1

normalized = defaultdict(lambda : defaultdict(int))
for year, topic_count in table.items():
	total = sum(topic_count.values())
	for topic, count in topic_count.items():
		normalized[year][topic] = float(count)/total

data  = pd.DataFrame(normalized)

print data
data.to_csv('topic_year_model.csv')

