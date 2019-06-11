from psyLex.psyLex.readDict import *
from psyLex.psyLex.wordCount import *

from sklearn.base import TransformerMixin

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
from collections import defaultdict
import struct

	
class EmbeddingVectorizer(object):
	def __init__(self, pre_trained, mean, input_file):
		self.pre_trained = pre_trained
		self.mean = mean
		self.input_file = input_file
	
	def fit(self, X, y=None):
		self.tokenized_x = []
		for doc in X:
			self.tokenized_x.append(word_tokenize(doc))
		if(self.pre_trained):
			'''
			with open(self.input_file,'rb') as lines:
				self.w2v = {line.split()[0].decode('utf-8'): np.array(line.split()[1:],dytpe=np.float32)
				for line in lines}
			'''
			self.w2v = {}
			all_words = set(w for words in self.tokenized_x for w in words)
			with open(self.input_file, "rb") as infile:
				for line in infile:
					parts = line.split()
					word = parts[0].decode('utf-8')
					if (word in all_words):
						nums=np.array(parts[1:], dtype=np.float32)
						self.w2v[word] = nums
		else:
			model = Word2Vec(self.tokenized_x, size=300)
			self.w2v = {w: vec for w, vec in zip(model.wv.index2word,model.wv.syn0)}

		self.dim = len(next(iter(self.w2v.values())))
		return self
		'''
		tfidf = TfidfVectorizer(analyzer=lambda x: x, decode_error = 'ignore')
		tfidf.fit(self.tokenized_x)
		# if a word was never seen - it must be at least as infrequent
		# as any of the known words - so the default idf is the max of 
		# known idf's
		max_idf = max(tfidf.idf_)
		self.word2weight = defaultdict(
		    lambda: max_idf,
		    [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
		return self
		'''
	
	def transform(self, X):
		self.tokenized_x = []
		for doc in X:
			self.tokenized_x.append(word_tokenize(doc))
		if(self.mean):
			X_new = np.array([
					np.mean([self.w2v[w]
						for w in words if w in self.w2v] or
						[np.zeros(self.dim)], axis=0)
					for words in self.tokenized_x
				])
		else:
			X_new = np.array([
					np.sum([self.w2v[w]
						for w in words if w in self.w2v] or
						[np.zeros(self.dim)], axis=0)
					for words in self.tokenized_x
				])
		print(X_new.shape)
		return X_new

class LIWC(TransformerMixin):
	def __init__(self, dictionary='resources/Dictionaries/LIWC2007_English131104.dic'):
		self.dictionary = dictionary
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		dictIn, catList = readDict(self.dictionary)
		vec = []
		X_LIWC = []
		for doc in X:
			doc = doc.replace('\n', ' ')
			doc = re.sub('[^ a-zA-Z]', '', doc)
			out = wordCount(doc, dictIn, catList)
			for k, v in out[0].items():
				vec.append(v)
			X_LIWC.append(vec)
			vec = []
		self.X_new = np.array(X_LIWC)
		print(self.X_new.shape)
		return self.X_new
	def get_params(self, deep=False):
		return {'LIWC':True}
		
class LemmaTokenizer(object):
	def __call__(self, text):
		lemmas = []
		lemmatizer = WordNetLemmatizer()
		for w in word_tokenize(text):
			lemma = lemmatizer.lemmatize(w)
			lemma = lemmatizer.lemmatize(lemma, pos='v')
			lemmas.append(lemma)
		return lemmas

class StemTokenizer(object):
	def __call__(self, text):
		stemmer = PorterStemmer()
		return [stemmer.stem(w) for w in word_tokenize(text)]
