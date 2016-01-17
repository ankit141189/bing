def about():
	print "Process the Microsoft Hackathon Files."

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
			document["title"] = fields[4]
			document["summary"] = fields[5].split(".")[:-1]
			documents += [document]
		return documents
		
def write_documents_as_string(docs, filename):
	out = open(filename, "w")
	for doc in docs:
		for sum in doc["summary"]:
			#out.write(doc["title"] + "\n")
			out.write(sum + "\n")
		
		
		
	
if __name__ == "__main__":
	about()
	documents_train = read_tsv_file("train.txt")	
	documents_test = read_tsv_file("test.txt")
	#print len(documents_test)
	write_documents_as_string(documents_train, "train_words.txt")
	write_documents_as_string(documents_test, "test_words.txt")