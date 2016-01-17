from scipy import sparse
from scipy.sparse import dok_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from scipy.io import mmread
import sys

def process_labels(labels):
	topic_a_labels = []
	topic_b_or_c_labels = []
	for index, label in enumerate(labels):
		if label == 0:
			topic_a_labels.append(1)
		elif label == 1:
			topic_b_or_c_labels.append((index, 1))
			topic_a_labels.append(0)
		else:
			topic_b_or_c_labels.append((index, 2))
			topic_a_labels.append(0)
	return topic_a_labels, topic_b_or_c_labels	

def sample(X, Y, size=500):
	if sparse.issparse(X):
		X = X.toarray()
	combined = np.append(X, np.matrix(Y).T, axis=1) 
	np.random.shuffle(combined)
	tail_size = -1 * size
	last_column = X.shape[1]
	training_labels = combined[:tail_size, last_column]
	training_data = combined[:tail_size, :-2]
	test_data = combined[tail_size:, :-2]
	actual_labels = combined[tail_size:, last_column]
	return training_data, np.ravel(training_labels), test_data, np.ravel(actual_labels)

training = open('summary_tfidf')
matrix = mmread(training).todok()

# NO_TRAINING_SAMPLES = 6000
# NO_OF_AUTHORS = 20000
# matrix = dok_matrix((NO_TRAINING_SAMPLES, NO_OF_AUTHORS), dtype=np.int)
# for line in training.readlines():
# 	values = line.rstrip().split()
#	print values
#	matrix[int(values[0]), int(values[1])] = float(values[2])

labels_file = open('training_labels')
labels = [int(x) for x in labels_file.readline().rstrip().split()]

training_matrix = matrix[:4498]
test_size = int(sys.argv[1])
topic_a_labels, topic_b_or_c_labels = process_labels(labels)
training_data, training_labels, test_data, actual_labels = sample(training_matrix, topic_a_labels, test_size)
classifier = LinearSVC()
classifier.fit(training_data, training_labels)

output = classifier.predict(test_data)
for index, predicted in enumerate(output):
	print '%s %s' % (predicted, actual_labels[index])

print classifier.score(test_data, actual_labels)
print confusion_matrix(actual_labels, output)

b_or_c_data = dok_matrix((len(topic_b_or_c_labels), training_matrix.shape[1]))
i = 0
for index, label in topic_b_or_c_labels:
	b_or_c_data[i, :] = training_matrix[index]
	i += 1
b_or_c_labels = [label for index, label in topic_b_or_c_labels]
training_data, training_labels, test_data, actual_labels = sample(b_or_c_data, b_or_c_labels, test_size)
classifier = GaussianNB()
classifier.fit(training_data, training_labels)

output = classifier.predict(test_data)
for index, predicted in enumerate(output):
	print '%s %s' % (predicted, actual_labels[index])

print classifier.score(test_data, actual_labels)
print confusion_matrix(actual_labels, output)
