from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.io as spio
import sys

content_type = [(5, 'summary')]
remove_words = open('remove_words').readline().rstrip().split()

def read_field(file, field):
	docs = []
	for line in file.readlines():
		fields = line.rstrip().split('\t')
		content = fields[field].replace('.', ' ').split()
		filtered_content = [word for word in content if word not in remove_words]
		docs.append(' '.join(filtered_content))
	return docs

training = open('BingHackathonTrainingData.txt')
test = open('BingHackathonTestData.txt')
for index, type in content_type:
	training.seek(0, 0)
	test.seek(0, 0)
	docs = read_field(training, index) + read_field(test, index)	
	tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
	data = tfidf.fit_transform(docs)
	tfidf_file = open(type + '_tfidf', 'w')
	spio.mmwrite(tfidf_file, data)
	word_map = {word : idx for idx, word in enumerate(tfidf.get_feature_names())}
	output_file = open(type + 'word_ids', 'w')
	print len(word_map)
	for key, value in word_map.items():
		output_file.write('%s %s\n' % (key, value + 1))

def calculate_bigrams(file):
	docs = []
	for line in file.readlines():
		fields = line.rstrip().split('\t')
		text = fields[5]
		bigram_doc = []
		for sentence in text.split('.'):
			words = sentence.split() 
 			bigram_doc.append(' '.join([x + "_" + y for x, y in zip(words[:-1], words[1:])]))
		
		docs.append(' '.join(bigram_doc))
	return docs

training.seek(0, 0)
test.seek(0, 0)
docs = calculate_bigrams(training) + calculate_bigrams(test)
print len(set(' '.join(docs).split()))
tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=0.005)
data = tfidf.fit_transform(docs)
print len(tfidf.get_feature_names())
bigram = open('bigrams', 'w')
spio.mmwrite(bigram, data)
