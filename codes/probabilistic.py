import gensim
import math
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
def getProbabilities(docs):
	words = []
	authors = []
	titles = []
	for doc in docs:
		for sum in doc["summary"]:
			for word in sum.split():
				words += [word]
		for author in doc["authors"]:
			authors += [author]
		for title in doc["title"]:
			titles += [title]

	all = words + titles + authors
	all = set(all)
	print len(all)
	word2index = {}
	i = 0
	for x in all :
		word2index[x] = i
		i += 1
		
	P_y_given_x = np.zeros([len(all), 3])
	N_y_and_x = np.zeros([len(all), 3])
	#Fill Up the matrix:
	for y in range(3):
		for doc in docs:
			for sum in doc["summary"]:
				for word in sum.split():
					if(doc["topic_id"] == y):
						N_y_and_x[word2index[word], y] += 1 
			for author in doc["authors"]:
				if(doc["topic_id"] == y):
					N_y_and_x[word2index[author], y] += 1 
			for title in doc["title"]:
				if(doc["topic_id"] == y):
					N_y_and_x[word2index[title], y] += 1 
	N_x = np.zeros(len(all))
	N_x = np.sum(N_y_and_x, axis = 1)
	P_y_given_x = (N_y_and_x.transpose() / N_x).transpose()
	return (word2index, P_y_given_x)
def test_documents(docs, word2index, P_y_given_x):
	correct = 0
	for doc in docs:
		sums = np.zeros(3);
		for sum in doc["summary"]:
			for word in sum.split():
				if word in word2index:
					sums +=  P_y_given_x[word2index[word],:]
		for author in doc["authors"]:
			if author in word2index:
				sums +=  P_y_given_x[word2index[author],:]
		for title in doc["title"]:
			if title in word2index:
				sums +=  P_y_given_x[word2index[title],:]
		#print np.argmax(sums), doc["topic_id"]
		if np.argmax(sums) == doc["topic_id"]:
			correct += 1
	print float(correct)/len(docs)
def read_tsv_file(filename):
		lines = open(filename).read().split("\n")[:-1]
		documents = []
		for line in lines:
			fields = line.split("\t")
			document = {}
			document["record_id"] = int(fields[0])
			document["topic_id"] = int(fields[1])
			document["publication_year"] = int(fields[2])
			document["authors"] = fields[3].split(";")
			document["title"] = fields[4].split()
			document["summary"] = fields[5].split(".")[:-1]
			documents += [document]
		return documents

	
if __name__ == "__main__":
	documents = read_tsv_file("train.txt")
	(word2index, P_y_given_x) = getProbabilities(documents[0:3000])
	test_documents(documents[3000:], word2index, P_y_given_x)
	#fit_multiclass_svm1(documents, idfs)
	#train_word2vec("words.txt")
	#test_conversion()
	#train_doc2vec("train_words.txt")
	#test_conversion_doc()