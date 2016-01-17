from scipy.sparse import dok_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import cross_validation
import numpy as np
import sys
from scipy import io as scio
from sklearn.ensemble import GradientBoostingClassifier

#First argument is the feature vector file
#Second argument is the training labels file
matrix = scio.mmread(sys.argv[1])
labels_file = open(sys.argv[2])

labels = [int(x) for x in labels_file.readline().rstrip().split()]
training_matrix = matrix[:4498]
sampled_training_matrix = training_matrix[:4000]
sampled_labels = labels[:4000]
#classifier = LinearSVC(C=0.08, class_weight={0:1.0, 1:1.0, 2:2.5})
#classifier = GradientBoostingClassifier()
classifier = MultinomialNB()
cv_score = cross_validation.cross_val_score(classifier, training_matrix, labels, cv=5)
print cv_score
classifier.fit(sampled_training_matrix, sampled_labels)
output = classifier.predict(training_matrix[4000:4498])
test_labels = labels[4000:4498]

print classifier.score(sampled_training_matrix, sampled_labels)
print confusion_matrix(sampled_labels, classifier.predict(sampled_training_matrix))

print classifier.score(training_matrix[4000:4498], test_labels)
print confusion_matrix(test_labels, output)
