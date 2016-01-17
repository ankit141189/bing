from scipy import sparse
from scipy.sparse import dok_matrix
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from scipy.io import mmread
import sys

def extend_training(X, Y):
	if sparse.issparse(X):
		X = X.toarray()
	extended_labels = list(Y)
	for index, label in enumerate(Y):
		if label == 2 and index % 2 == 0:
			X = np.append(X, X[index, :], axis=0)
			extended_labels.append(label)
	return X, extended_labels
		
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

labels_file = open('topic_training_labels')
labels = [int(x) for x in labels_file.readline().rstrip().split()]

training_matrix = matrix
print matrix.shape
print len(labels)
test_size = int(sys.argv[1])
training_data, training_labels, test_data, actual_labels = sample(training_matrix, labels, test_size)
# training_data, training_labels = extend_training(training_data, training_labels)
classifier = LinearSVC(C=0.5, class_weight={1:1.0, 2:3.0})
classifier.fit(training_data, training_labels)
output = classifier.predict(test_data)
for index, predicted in enumerate(output):
#	print '%s %s' % (predicted, actual_labels[index])
	pass

print classifier.score(test_data, actual_labels)
print confusion_matrix(actual_labels, output)

output = classifier.predict(training_data)
for index, predicted in enumerate(output):
#	print '%s %s' % (predicted, training_labels[index])
	pass

print classifier.score(training_data, training_labels)
print confusion_matrix(training_labels, output)
