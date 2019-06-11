import nltk.corpus
#import autosklearn.classification
from sklearn.svm import LinearSVC
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader

from corpus import Corpus
from strategies import RepresentationSearch
from joblib import load

class Autotext:
	
	def __init__(self, strategy='nearest', limit_memory = True, verbose = True):
		self.iters = 0
		self.strategy = strategy
		self.limit_memory = limit_memory
		self.verbose = verbose
		#self.automl = autosklearn.classification.AutoSklearnClassifier(ensemble_memory_limit = 18000, ml_memory_limit = 111000, n_jobs = 2)
		self.automl = LinearSVC()
		self.y_test = []
		
	def train(self, corpus_path, skip = ''):
		if (self.iters == 0):
			if(self.verbose):
				print('Extracting meta-features...')
			self.meta_features = self._extract_meta_data(corpus_path)
			if(self.verbose):
				print('Searching representation...')
			self.rep_id = self._find_representation(self.meta_features, self.strategy, exclude = skip)
		
		self.representation = load('../metadata/'+str(self.rep_id[self.iters])+'.joblib')
		#if(self.verbose):
		#	print(str(self.iters) + ' Representation id: '+str(self.rep_id[self.iters]))
		with open('../results/log', 'a+') as f_log:
			f_log.write('Rep id for '+corpus_path+ ': '+str(self.rep_id[self.iters])+'\n')
		
		documents, self.y_train = self._read_corpus(self.db, corpus_path)
		
		self.X_train = self.representation.fit_transform(documents)	
		print('Samples: ')
		print(len(self.X_train))
		#print(self.X_train)
		#print('Classes :')
		#print(self.y_train)
		
		#if(self.verbose):
			#print('Searching classification model (AutoSklearn)...')
		
		self.automl.fit(self.X_train, self.y_train)
		
		self.iters += 1
		
		return self
	
	def predict(self, test_path):
		documents, self.y_test = self._read_corpus(CategorizedPlaintextCorpusReader(test_path, r'.*\.txt', cat_pattern=r'(\w+)/*'),test_path)
		self.X_test = self.representation.transform(documents)
		return self.automl.predict(self.X_test)
		
	def _extract_meta_data(self, corpus_path):
		self.db = CategorizedPlaintextCorpusReader(corpus_path, r'.*\.txt', cat_pattern=r'(\w+)/*')
		new_corpus = Corpus(self.db, corpus_path, self.limit_memory, self.verbose)
		return new_corpus.get_meta_features()
		
	def _find_representation(self, meta_data, strategy, exclude = ''):
		finder = RepresentationSearch(meta_data, exclude)
		if (strategy == 'nearest'):
			representation = finder.nearest_predict()
		elif (strategy == 'classif'):
			representation = finder.classif_predict()
		elif (strategy == 'regression'):
			representation = finder.regressor_predict()
		elif (strategy == 'rank'):
			representation = finder.rank_predict()
		elif (strategy == 'top5'):
			representation = finder.top5()
		elif (strategy == 'all'):
			representation = finder.all_r()
		return representation
	
	def _read_corpus(self, corpus, path):
		#Lists for sklearn
		documents = []
		targets = []
		j = 0
		#print('Reading files')
		try:
			for cat in corpus.categories():
				for doc in corpus.fileids(cat):
					documents.append(corpus.raw(doc))
					targets.append(j)
				j+=1
		except:
			j = 0
			for cat in corpus.categories():
				for doc in corpus.fileids(cat):
					raw_document = open(path+doc, errors='ignore')
					documents.append(raw_document.read())
					targets.append(j)
					raw_document.close()
				j+=1
		return documents, targets
		
