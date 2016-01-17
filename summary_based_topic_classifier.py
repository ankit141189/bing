from scipy import sparse
from scipy.sparse import dok_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from scipy.io import mmread
import sys
import random
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

def flatten(l):
	return [int(x) for x in np.ravel(l)]

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
	return training_data, flatten(training_labels), test_data, flatten(actual_labels)

summary_data = open('summary_tfidf')
extra_features_data = open('extra_counts')
bigrams_data = open('bigrams')
matrix = mmread(summary_data)
extra_features = mmread(extra_features_data)
bigrams = mmread(bigrams_data)
print extra_features.shape
print matrix.shape
print bigrams.shape
#matrix = np.append(extra_features, matrix.toarray(), axis=1)
matrix = np.append(np.append(extra_features, matrix.toarray(), axis=1), bigrams.toarray(), axis=1) 
print matrix.shape

labels_file = open('topic_training_labels')
labels = [int(x) for x in labels_file.readline().rstrip().split()]

training_matrix = matrix[:4498]
test_size = int(sys.argv[1])
training_data, training_labels, test_data, actual_labels = sample(training_matrix, labels, test_size)
classifier = MultinomialNB() #LinearSVC(C=0.5, class_weight={0:1.0, 1:1, 2:2.0})
#nbrs = KNeighborsClassifier(n_neighbors=3).fit(training_data, training_labels)
classifier.fit(training_data, training_labels)
output = classifier.predict(test_data)
#decision = classifier.decision_function(test_data)
#for scores, label, actual in zip(decision.tolist(), output, actual_labels):
#	print [round(value, 4) for value in scores], label, actual
combined_output = output
#for index, predicted in enumerate(output):
#	print '%s %s' % (predicted, actual_labels[index])
#	if not predicted == 0:
#		combined_output += (nbrs.predict(test_data[index]).tolist()) 
#	else:
#		combined_output.append(predicted)

print metrics.accuracy_score(actual_labels, combined_output)
print confusion_matrix(actual_labels, combined_output)

