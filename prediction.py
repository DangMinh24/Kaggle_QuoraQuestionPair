import pandas as pd
import gensim
import gensim.models.doc2vec
from gensim.models import Doc2Vec
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import sys

p_stemmer = PorterStemmer()
en_stop = get_stop_words('en')
tokenizer = RegexpTokenizer(r'\w+')

model = Doc2Vec.load("my_model3.doc2vec")

def clean_text(sentence):
	# str_sentence = str(sentence)
	# unicode_sentence = unicode(sentence,errors="ignore")
	raw = sentence.lower()
	tokens = tokenizer.tokenize(raw)
	stopped_tokens = [i for i in tokens if not i in en_stop]
	texts = [p_stemmer.stem(i) for i in stopped_tokens]
	return " ".join(texts)

def read_data(file):
	df = pd.read_csv(file,encoding = 'utf8')
	df.drop_duplicates(inplace=True)
	df.dropna(inplace=True)
	return df 

data_test= read_data("test.csv")

print data_test.shape
for index, row in data_test.iterrows():

	q1 = row["question1"]
	q2 = row["question2"]
	# try:
	q1 = clean_text(q1)		
	q2 = clean_text(q2)
	# except:
	# 	print "Error : " + q1
	# 	print "Error : " + q2
	# 	continue
	similarity = model.docvecs.similarity_unseen_docs(model,q1.split(" "),q2.split(" "))
	# print abs(similarity)
	with open("result2.csv","a") as f:
		# sim = 1 if similarity > 0.5 else 0
		f.write(str(row["test_id"])+","+str(abs(similarity)) +"\n")