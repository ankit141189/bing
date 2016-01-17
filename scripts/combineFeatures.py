import sys
import numpy as np
from scipy import io as spio
from scipy import sparse as spsp

num_args = len(sys.argv)

R = np.random.randn(5623,0)
for i in range(1, num_args - 1):
	X = spio.mmread(sys.argv[i])
	print R.shape, X.shape
	if(spsp.issparse(X)):
		print X.toarray().shape
		R = np.append(R, X.toarray(), 1)
	else:
		R = np.append(R, X, 1)

print R.shape

spio.mmwrite(sys.argv[num_args - 1], R)
