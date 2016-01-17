import sys
for line in open(sys.argv[1],'r'):
	toks = line.rstrip().split("\t")
	print toks
