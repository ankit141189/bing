from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import interactive

interactive(True)


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
		table[int(authors[auth])][fields[2]] += 1

normalized = defaultdict(lambda : defaultdict(float))
for auth, year_count in table.items():
	total = sum(year_count.values())
	for year, count in year_count.items():
		normalized[auth][year] = float(count)/total 

data = pd.DataFrame(normalized)

print data
data.to_csv('year_author_model.csv')
