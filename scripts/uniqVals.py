import sys

#Argument 1 is the file name
input_file = sys.argv[1]
#Argument 2 is the column number starting from 0
col_num = sys.argv[2]
#Argument 3 is the file to print the vocab to
output_file = sys.argv[3]

vocab = {}

for line in open(sys.argv[1], 'r'):
	toks  = line.rstrip().split('\t')
	vals = []
	if ";" in toks[col_num]:
		vals = toks[col_num].trim().split(';')
	elif "." in toks[col_num]:
		vals = toks[col_num].trim().replace(',',' ').split(' ')
	else:
		vals = toks[col_num].trim().split(' ')
	
	for val in vals:
		if val not in vocab:
			vocab[val] = 0
		vocab[val] += 1
	

for val, count in vocab.iteritems()
