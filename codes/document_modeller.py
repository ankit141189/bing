import gensim
import math
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for uid, line in enumerate(open(self.filename)):
			yield gensim.models.doc2vec.LabeledSentence(words=line.split(), tags=['SENT_%s' % uid])

def about():
	print "Train Word2vec and Doc2vec."
		
def train_word2vec(train_file):
	model = gensim.models.Word2Vec(size=50)
	sentences = gensim.models.word2vec.LineSentence(train_file)
	model.build_vocab(sentences)
	model.train(sentences)
	model.save("train_word2vec.model")

def test_conversion(document):
	model = gensim.models.Word2Vec.load("train_word2vec.model")

def test_conversion_doc():
	model = gensim.models.Doc2Vec.load("train_doc2vec.model")
	print model["SENT_0"]
	
def fit_multiclass_svm(documents, idfs):
	model = gensim.models.Word2Vec.load("train_word2vec.model")
	dim = 50;
	X = np.zeros([4000, dim]);
	X_test = np.zeros([490, dim]);
	y = np.zeros(4000);
	y_test = np.zeros(490);
	i = 0
	for doc in documents[:4000]:
		x = np.zeros(dim)
		count = 0
		for sent in doc["summary"]:
			for word in sent.split():
				if word in model:
					x = x + (idfs[word] * model[word])
					count += 1
		X[i, :] = x/count
		y[i] = doc["topic_id"]
		i = i + 1;
	svm_model = OneVsRestClassifier(LinearSVC(random_state=0, C = 1)).fit(X, y)
	
	
	i = 0
	for doc in documents[4000:4490]:
		x = np.zeros(dim)
		count = 0
		for sent in doc["summary"]:
			for word in sent.split():
				if word in model:
					x = x + (idfs[word] * model[word])
					count += 1
		X_test[i, :] = x/count
		y_test[i] = doc["topic_id"]
		i = i + 1;
	print svm_model.score(X_test, y_test)
	
def fit_multiclass_svm1(documents, idfs):
	model = gensim.models.doc2vec.Doc2Vec.load("train_doc2vec.model")
	X = np.zeros([4000, 300]);
	X_test = np.zeros([490, 300]);
	y = np.zeros(4000);
	y_test = np.zeros(490);
	i = 0
	for doc in documents[:4000]:
		x = np.zeros(300)
		count = 0
		for sent in doc["summary"]:
			for word in sent.split():
				if word in model:
					x = x + (idfs[word] * model[word])
					count += 1
		X[i, :] = x/count
		y[i] = doc["topic_id"]
		i = i + 1;
	svm_model = OneVsRestClassifier(svm.SVC(kernel='poly', gamma=2)).fit(X, y)
	
	
	i = 0
	for doc in documents[4000:4490]:
		x = np.zeros(300)
		count = 0
		for sent in doc["summary"]:
			for word in sent.split():
				if word in model:
					x = x + (idfs[word] * model[word])
					count += 1
		X_test[i, :] = x/count
		y_test[i] = doc["topic_id"]
		i = i + 1;
	print svm_model.score(X_test, y_test)
	
def read_tsv_file(filename):
		lines = open(filename).read().split("\n")[:-1]
		documents = []
		for line in lines:
			fields = line.split("\t")
			document = {}
			document["record_id"] = int(fields[0])
			document["topic_id"] = int(fields[1])
			document["publication_year"] = int(fields[2])
			document["authors"] = fields[3].split(";")
			document["title"] = fields[4].split()
			document["summary"] = fields[5].split(".")[:-1]
			documents += [document]
		return documents

def get_idf(docs):
	model = gensim.models.Word2Vec.load("train_word2vec.model")
	words = []
	idfs = {}
	for word in model.vocab:
		words += [word]
	words_set = set(words)
	for doc in docs:
		doc_words = []
		for sent in doc["summary"]:
			for word in sent.split():
				doc_words += [word]
		doc_words = set(doc_words)
		
		for dw in doc_words:
			if dw in idfs:
				idfs[dw] += 1;
			else:
				idfs[dw] = 1;
	N = len(docs)
	for w in idfs:
		idfs[w] = math.log(1 + (N/idfs[w]))
	return idfs

def train_doc2vec(train_file):
	model = gensim.models.doc2vec.Doc2Vec()
	sentences = LabeledLineSentence(train_file)
	model.build_vocab(sentences)
	#print model.vocab
	model.train(sentences)
	model.save("train_doc2vec.model")
	
if __name__ == "__main__":
	about()
	documents = read_tsv_file("train.txt")
	idfs = get_idf(documents)
	fit_multiclass_svm(documents, idfs)
	#fit_multiclass_svm1(documents, idfs)
	#train_word2vec("words.txt")
	#test_conversion()
	#train_doc2vec("train_words.txt")
	#test_conversion_doc()