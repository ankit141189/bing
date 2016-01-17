from scipy.sparse import dok_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import numpy as np

training = open('author_features')
NO_TRAINING_SAMPLES = 6000
NO_OF_AUTHORS = 10000
matrix = dok_matrix((NO_TRAINING_SAMPLES, NO_OF_AUTHORS), dtype=np.int)
for line in training.readlines():
	values = line.rstrip().split()
	matrix[int(values[0]), int(values[1])] = 1

labels_file = open('training_labels')
labels = labels_file.readline().rstrip().split()
training_matrix = matrix[:4498]
sampled_training_matrix = training_matrix[:4000]
sampled_labels = labels[:4000]
classifier = SVC()
classifier.fit(sampled_training_matrix, sampled_labels)
output = classifier.predict(training_matrix[4000:4498])
test_labels = labels[4000:4498]
for index, predicted in enumerate(output):
	print '%s %s' % (predicted, test_labels[index])

print classifier.score(training_matrix[4000:4498], test_labels)

print confusion_matrix(test_labels, output)
