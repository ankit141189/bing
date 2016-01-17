from collections import defaultdict
def read_authors(file):
	all_authors = []
	for line in file.readlines():
		fields = line.rstrip().split('\t')
		authors = fields[3].split(';')
		for author in authors:
			all_authors.append(author)
	return all_authors

def filter_authors(authors):
	freq_dist = defaultdict(int)
	for author in authors:
		freq_dist[author] += 1
	return [author for author, count in freq_dist.items() if count > 1]

training = open('BingHackathonTrainingData.txt')
test = open('BingHackathonTestData.txt')
all_authors = set(filter_authors(read_authors(training) + read_authors(test)))
authors_file = open('freq_authors', 'w')
for index, author in enumerate(all_authors):
	authors_file.write('%s:%s\n' % (author, index))

