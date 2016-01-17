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
print "Data loaded"
labels_file = open(sys.argv[2])

labels = [float(x) for x in labels_file.readline().rstrip().split()]
training_matrix = matrix[:4498]
sampled_training_matrix = training_matrix[:4000]
test_matrix = training_matrix[4000:4498]
sampled_labels = labels[:4000]
test_lables = labels[4000:4498]

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf = svr_rbf.fit(sampled_training_matrix, sampled_labels).predict(test_matrix)
#y_rbf = [0.0] * len(test_lables)
#y_lin = svr_lin.fit(sampled_training_matrix, sampled_labels).predict(test_matrix)
#y_poly = svr_poly.fit(sampled_training_matrix, sampled_labels).predict(test_matrix)

y_rbf_int = [ int(round(x)) for x in y_rbf]

for i in range(len(y_rbf_int)):
	print str(test_lables[i])+' '+str(y_rbf_int[i])

rmse_rbf = skmets.mean_squared_error(test_lables, y_rbf_int)
#rmse_lin = skmets.mean_squared_error(test_lables, y_lin)
#rmse_poly = skmets.mean_squared_error(test_lables, y_poly)	

print rmse_rbf
#print rmse_lin
#print rmse_poly


