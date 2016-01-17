import sys
import numpy as NP
from scipy import sparse as scsp
from scipy import io as scio

author_file_name = sys.argv[1]
num_authors = int(sys.argv[2])
output_file_name = sys.argv[3]

X = NP.zeros([5623, num_authors])

for line in open(author_file_name):
	toks = line.strip().split(' ')
	X[int(toks[0]) - 1, int(toks[1]) - 1] = 1

print scsp.csr_matrix(X)
scio.mmwrite(output_file_name, X)
