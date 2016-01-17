from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.io as spio
import sys

content_type = [(4, 'title'), (5, 'summary')]

def read_field(file, field):
	docs = []
	for line in file.readlines():
		fields = line.rstrip().split('\t')
		content = fields[field]
		docs.append(content)
	return docs

training = open('newData')
for index, type in content_type:
	training.seek(0, 0)
	docs = read_field(training, index)
	tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
	data = tfidf.fit_transform(docs)
	tfidf_file = open(type + '_tfidf', 'w')
	spio.mmwrite(tfidf_file, data)
	word_map = {word : idx for idx, word in enumerate(tfidf.get_feature_names())}
	output_file = open(type + 'word_ids', 'w')
	for key, value in word_map.items():
		output_file.write('%s %s\n' % (key, value + 1))
