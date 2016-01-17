from collections import defaultdict
import math
topic_word_count = defaultdict(lambda : defaultdict(int))
bigram_docs = []
sentences = []
for line in open('BingHackathonTrainingData.txt').readlines():
	values = line.rstrip().split('\t')
	topic = int(values[1])
	text = values[5]
	sentences += text.rstrip().split('.')
	for word in text.replace('.', ' ').split():
		topic_word_count[word][topic] += 1

for sent in sentences:
	print sent

normalized_topic_word_count = defaultdict(lambda : defaultdict(int))
for word, topic_count in topic_word_count.items():
	total = sum(topic_count.values())
	for topic, count in topic_count.items():
		normalized_topic_word_count[word][topic] = float(count)/total

for word, topic_dist in normalized_topic_word_count.items():
	max_value = max(topic_dist.values())
	next_value = sum(topic_dist.values()) - max_value - min(topic_dist.values())
	if max_value - next_value < 0.2 :
		#print word
		pass
 

