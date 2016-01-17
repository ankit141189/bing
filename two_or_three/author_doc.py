from collections import defaultdict
import pandas as pd

authors_file = open('freq_authors')
authors = {}
for line in authors_file.readlines():
	pair = line.rstrip().split(':')
	authors[pair[0]] = pair[1]

docid = 1
docs = []
def gen_doc_vector(file): 
	global docs
	global docid
	for line in file.readlines():
		fields = line.rstrip().split('\t')
		auths = fields[3].split(';')
		for auth in auths:
			if auth in authors:		
				docs.append((docid, authors[auth]))
		docid += 1
	
training_file = open('BingHackathonTrainingData.txt')
gen_doc_vector(training_file)
training_file.close()

test_file = open('BingHackathonTestData.txt')
gen_doc_vector(test_file)
test_file.close()

auth_out = open('filtered_author_features', 'w')
for docid, auth in docs:
	auth_out.write('%d %s\n' % (docid, auth))

