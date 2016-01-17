from collections import defaultdict
import pandas as pd

training_file = open('BingHackathonTrainingData.txt')
authors_file = open('authors')
authors = {}
for line in authors_file.readlines():
	pair = line.rstrip().split(':')
	authors[pair[0]] = pair[1]

table = defaultdict(lambda : defaultdict(int))
for line in training_file.readlines():
	fields = line.rstrip().split('\t')
	auths = fields[3].split(';')
	for auth in auths:		
		table[int(authors[auth])][fields[1]] += 1

normalized = defaultdict(lambda : defaultdict(float))
for auth, topic_count in table.items():
	total = sum(topic_count.values())
	for topic, count in topic_count.items():
		normalized[auth][topic] = float(count)/total 

data = pd.DataFrame(normalized)

print data
data.to_csv('topic_author_model.csv')

