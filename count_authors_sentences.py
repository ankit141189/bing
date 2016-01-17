from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.io as spio
import sys
import numpy as np

def read_field(file):
	docs = []
	for line in file.readlines():
		fields = line.rstrip().split('\t')
		no_sentences = len(fields[5].split('.'))
		no_authors = len(fields[3].split(';'))
		title_length = len(fields[4].split())
		docs.append([no_sentences, no_authors, title_length])
	return docs

training = open('BingHackathonTrainingData.txt')
test = open('BingHackathonTestData.txt')
docs = read_field(training) + read_field(test)	
total = [max(x) for x in zip(*docs)]
docs = [[round(float(x)/total[i],5) for i, x in enumerate(l)] for l in docs]
matrix = np.matrix(docs)
counts_file = open('extra_counts', 'w')
spio.mmwrite(counts_file, matrix)
for l in docs:
	print l
