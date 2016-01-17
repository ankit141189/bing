from scipy import sparse
from scipy.sparse import dok_matrix
from sklearn.svm import LinearSVR
from sklearn import metrics 
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

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

training = open('author_features')
NO_TRAINING_SAMPLES = 6000
NO_OF_AUTHORS = 10000
matrix = dok_matrix((NO_TRAINING_SAMPLES, NO_OF_AUTHORS), dtype=np.int)
for line in training.readlines():
	values = line.rstrip().split()
	matrix[int(values[0]), int(values[1])] = 1

labels_file = open('year_training_labels')
labels = [int(x) for x in labels_file.readline().rstrip().split()]

training_matrix = matrix[:4498]
training_data, training_labels, test_data, actual_labels = sample(training_matrix, labels)
classifier = LinearSVR()
classifier.fit(training_data, training_labels)
output = classifier.predict(test_data)
for index, predicted in enumerate(output):
	print '%s %s' % (predicted, actual_labels[index])

print metrics.explained_variance_score(actual_labels, output)
