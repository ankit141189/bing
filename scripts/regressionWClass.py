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
from sklearn.svm import SVR
import math
from sklearn import metrics as skmets

#First argument is the feature vector file
#Second argument is the training labels file
matrix = scio.mmread(sys.argv[1])
#matrix = dok_matrix((5623, 21271))
#matrix = matrix.toarray()
print "Data loaded"
labels_file = open(sys.argv[2])

labels = [float(x) for x in labels_file.readline().rstrip().split()]
training_matrix = matrix[:4498]
sampled_training_matrix = training_matrix[:4000]
test_matrix = training_matrix[4000:4498]
sampled_labels = labels[:4000]
test_lables = labels[4000:4498]

y_classification = labels[:4000]
num_class_1 = 0
num_class_2 = 0
for i in range(len(y_classification)):
	if y_classification[i] >= 2049:
		y_classification[i] = 1
		num_class_2 += 1
	else:
		y_classification[i] = 0
		num_class_1 += 1


print num_class_1
print num_class_2

num_cols = len(sampled_training_matrix[0])
X4348 = np.ndarray((num_class_1, num_cols))
Y4348 = np.ndarray((num_class_1, 1))
p4348 = 0
X4954 = np.ndarray((num_class_2, num_cols))
Y4954 = np.ndarray((num_class_2, 1))
p4954 = 0

for i in range(len(y_classification)):
	if y_classification[i] == 0:
		X4348[p4348:] = sampled_training_matrix[i]
		Y4348[p4348:] = sampled_labels[i]
		p4348 += 1
	else:
		X4954[p4954:] = sampled_training_matrix[i]
		Y4954[p4954:] = sampled_labels[i]
		p4954 += 1
	print str(i) + '\r',

print "Data split for classification and regression"

print X4348.shape
print Y4348.shape
print X4954.shape
print Y4954.shape

svr_4348 = SVR(kernel='rbf', C=1e3, gamma=0.1).fit(X4348, Y4348)
svr_4954 = SVR(kernel='rbf', C=1e3, gamma=0.1).fit(X4954, Y4954)

print "Regression fit"

classifier = LinearSVC()
classifier.fit(sampled_training_matrix, y_classification)
y_classification_pred = classifier.predict(test_matrix)

print "Classification done"

y_final = np.ndarray((len(test_lables),1))
for i in range(len(y_classification_pred)):
	if y_classification_pred[i] == 0:
		y_final[i:] = svr_4348.predict(np.matrix(test_matrix[i]))
	else:
		y_final[i:] = svr_4954.predict(np.matrix(test_matrix[i]))

y_rbf_int = [ int(round(x)) for x in y_final]

print len(test_lables)
print len(y_classification_pred.tolist())
print len(y_final.tolist())
print len(y_rbf_int)

for i in range(len(y_rbf_int)):
	print str(test_lables[i])+' '+str(y_rbf_int[i])+' '+str(y_classification_pred[i])

rmse_rbf = skmets.mean_squared_error(test_lables, y_rbf_int)
#rmse_lin = skmets.mean_squared_error(test_lables, y_lin)

#rmse_poly = skmets.mean_squared_error(test_lables, y_poly)	

print rmse_rbf
#print rmse_lin
#print rmse_poly


