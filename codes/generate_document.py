import networkx as nx
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
 
def textrank(sentences):
	bow_matrix = CountVectorizer().fit_transform(sentences)
	normalized = TfidfTransformer().fit_transform(bow_matrix)
	similarity_graph = normalized * normalized.T
	nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
	scores = nx.pagerank(nx_graph)
	return sorted(((scores[i],i,s) for i,s in enumerate(sentences)),
                  reverse=True)
		
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
		
import random

class Markov(object):
	
	def __init__(self, titles):
		self.cache = {}
		self.titles = titles
		self.words = self.title_to_words()
		self.word_size = len(self.words)
		self.database()
		
	
	def title_to_words(self):
		words = self.titles.strip().split()
		return words
		
	
	def triples(self):
		""" Generates triples from the given data string. So if our string were
				"What a lovely day", we'd generate (What, a, lovely) and then
				(a, lovely, day).
		"""
		
		if len(self.words) < 3:
			return
		
		for i in range(len(self.words) - 2):
			yield (self.words[i], self.words[i+1], self.words[i+2])
			
	def database(self):
		for w1, w2, w3 in self.triples():
			key = (w1, w2)
			if key in self.cache:
				self.cache[key].append(w3)
			else:
				self.cache[key] = [w3]
				
	def generate_markov_text(self, size=25):
		seed = self.words.index('<BOT>')
		seed_word, next_word = self.words[seed], self.words[seed+1]
		w1, w2 = seed_word, next_word
		gen_words = []
		for i in xrange(size):
			gen_words.append(w1)
			w1, w2 = w2, random.choice(self.cache[(w1, w2)])
		gen_words.append(w2)
		trimmed = []
		for word in gen_words:
			if word == "<EOT>":
				break;
			else:
				trimmed += [word]
		return ' '.join(trimmed[1:])
			
def generate_document(documents, k = 5):
	out_file = open("generated.txt", "w")
	for record_id in range(k):
		Y_ind = random.randint(0, 100)
		y = 0
		if Y_ind < 50:
			y = 0
		elif Y_ind < 80:
			y = 1
		else :
			y = 2
		#y =  random.randint(0, 2)
		total = len(documents)
		limit = 10
		count = 0
		year = 0
		sampled = set([])
		sentences = []
		titles = ""
		authors = []
		author_lengths = np.zeros(len(documents))
		for (ii,doc) in enumerate(documents):
			author_lengths[ii] = len(doc["authors"])
	
		mu = np.mean(author_lengths)
		sigma = np.std(author_lengths, ddof=1)
		
		while count < limit :
			sample = random.randint(0, total-1)	
			if(sample not in sampled) and (documents[sample]['topic_id'] == y):
				sampled.add(sample)
				count += 1
				sentences += documents[sample]['summary']
				year += documents[sample]['publication_year']
				titles += (" <BOT> " + " ".join(documents[sample]['title']) + " <EOT> ")
				authors += documents[sample]['authors']
		
		ranked = sorted(textrank(sentences)[0:8], key = lambda x: x[1])
		year = (year/limit)
		markov = Markov(titles)
		author_length = np.random.normal(mu, sigma, 1)
		sampled_authors = set([])
		
		count = 0
		while count < author_length:
			sample_index = random.randint(0, len(authors) - 1)
			sample = authors[sample_index]	
			if(sample not in sampled_authors):
				count += 1
				sampled_authors.add(authors[sample_index])
				
				
		sampled_authors = ";".join(sampled_authors)
		title = markov.generate_markov_text()
		summary = []
		for sent in ranked:
			summary += [sent[2]]
		summary = ".".join(summary)
		out_string = ""
		out_string = str(record_id) + "\t" + str(y) + "\t" + str(year) + "\t" + sampled_authors + "\t" + title + "\t" + summary + "\n"
		out_file.write(out_string)
		print out_string
		
if __name__ == "__main__":
	documents = read_tsv_file("train.txt")
	generate_document(documents, k = 5)
	
	