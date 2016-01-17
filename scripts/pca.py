import sys
from scipy import io as sp
from sklearn.decomposition import PCA
from scipy import sparse as scsp

X = sp.mmread(sys.argv[1])
print "Data read"
pca = PCA(n_components=int(sys.argv[2]))
print "PCA loaded"
if scsp.issparse(X):
	Xtrans = pca.fit_transform(X.toarray())
	print "Data transformed"
	sp.mmwrite(sys.argv[1]+'_pruned',Xtrans)
	print "Data written"
else:
	Xtrans = pca.fit_transform(X)
        print "Data transformed"
        sp.mmwrite(sys.argv[1]+'_pruned',Xtrans)
        print "Data written"

