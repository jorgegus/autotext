import nltk.corpus
import nltk.probability
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

import textstat

import re
import subprocess
import os
import numpy as np
import math
import csv
from collections import Counter

from scipy.stats import entropy
from scipy.stats import tstd
from scipy.stats import describe

#import textmining

class Corpus:
	def __init__(self, corpus, path, limit_memory = True, verbose = True):
		self.path = path
		self.cor = corpus
		self.limit = limit_memory
		self.verbose = verbose
		self.docs_limit = 500000
		
		#General meta-features
		self.number_of_documents = len(corpus.fileids())
		self.number_of_categories = len(corpus.categories())
		self.word_length = 0
		
		self.per_cat_limit = 90000#//self.number_of_categories
		
		#Hardness
		self.imbalance = 0 #Class imbalance
		self.SEM = 0 #Stylometric Evaluation Measure
		self.SVB = 0 #Supervised Vocabulary Based
		self.UVB = 0 #Unsupervised Vocabulary Based
		self.MRH_J = 0 #Macro-average Relative Hardness-Jaccard
		self.VL = 0 #Vocabulary Length
		self.VDR = 0 #Vocabulary Document Ratio
		
		#BoW Statistics
		self.pcac_bow = 0
		self.pca_singular_sum_bow = 0
		self.pca_explained_ratio_bow = 0
		self.pca_explained_var_bow = 0
		self.pcap_bow = 0
		self.pcaminmax_bow = 0
		self.pcaavg_bow = 0
		self.pcavar_bow = 0
		self.pcaskw_bow = 0
		self.pcakur_bow = 0
		self.onenn = 0 #Data sparsity (1NN accuracy)
		self.tree = 0 #Data separability (DT)
		self.lin = 0 #Linear separability (LinearDiscriminantAnalysis)
		self.nb = 0 #Feature independence (NB)
		self.zero_pct = 0

		#Documents per category statistics
		self.docs_per_category = []
		self.minmax_dpc = 0
		self.avg_dpc = 0
		self.sd_dpc = 0
		self.skw_dpc = 0
		self.kur_dpc = 0
		self.ratio_avg_sd_dpc = 0
		self.entropy_dpc = 0

		#Words per document statistics
		self.words_per_doc = []
		self.minmax_wpd = 0
		self.avg_wpd = 0
		self.sd_wpd = 0
		self.skw_wpd = 0
		self.kur_wpd = 0
		self.ratio_avg_sd_wpd = 0
		self.entropy_wpd = 0
		
		#Vocabulary statistics
		#Most frequent word frequency, average word frequency, etc
		self.size_voc = 0
		self.minmax_voc = 0
		self.avg_voc = 0
		self.sd_voc = 0
		self.skw_voc = 0
		self.kur_voc = 0
		self.ratio_avg_sd_voc = 0
		self.entropy_voc = 0
		
		#POS tagging statistics / lexical features
		self.adj_pct = 0 #Percentage of adjectives in corpus
		self.adp_pct = 0 #Adpositions percentage
		self.adv_pct = 0 #Adverbs percentage
		self.conj_pct = 0 #Conjunctions percentage
		self.det_pct = 0 #Articles percentage
		self.noun_pct = 0 #Nouns percentage
		self.num_pct = 0 #Numerals percentage
		self.prt_pct = 0 #Particles percentage
		self.pron_pct = 0 #Pronouns percentage
		self.verb_pct = 0 #Verbs percentage
		self.mark_pct = 0 #Punctuation marks percentage
		self.other_pct = 0 #Untagged words percentage
		
		#Syntactical Complexity features (L2SCA)
		#All features are calculated per document and averaged
		self.complexity_features = []
		for i in range(23):
			self.complexity_features.append(0)
		self.words = 0 #W
		self.sentences = 0 #S
		self.verb_phrases = 0 #VP
		self.clauses = 0 #C
		self.T_units = 0 #T
		self.dependent_clauses = 0 #DC
		self.complex_T_units = 0 #CT
		self.coordinate_phrases = 0 #CP
		self.complex_nominals = 0 #CN
		self.sentences_length = 0
		self.T_unit_length = 0
		self.clause_length = 0
		self.C_per_S = 0
		self.VP_per_T = 0
		self.C_per_T = 0
		self.DC_per_C = 0
		self.DC_per_T = 0
		self.T_per_S = 0
		self.CT_per_T = 0
		self.CP_per_T = 0
		self.CP_per_C = 0
		self.CN_per_T = 0
		self.CP_per_C = 0
		
		#Textstat measures
		self.reading_ease = 0
		self.smog_index = 0
		self.kincaid_grade = 0
		self.coleman_liau_index = 0
		self.readability_index = 0
		self.dale_chall_score = 0
		self.difficult_words = 0
		self.linsear_formula = 0
		self.guning_fog = 0
		self.school_grade = 0
		
		self.calculate_meta_features()
		
	def analyze_complexity(self, doc_file):
		temp_file = '../src/tmp/syntactic_complexity.dat'
		command = 'sh execute_L2SCA.sh ' + doc_file + ' ' +temp_file
		os.system(command)	
		with open(temp_file) as complexity_file:
			complexity_reader = csv.reader(complexity_file)
			features = next(complexity_reader)
			features = next(complexity_reader)
			for i in range(1,24):
				self.complexity_features[i-1] += float(features[i])
			
				
	def calculate_meta_features(self):
		endc = self.number_of_documents/self.number_of_categories
		category_vocabulary_size = []
		category_vocabulary = []
		all_words = []
		document_vocabulary_size = []
		raw_documents_path = []
		target_documents = []
		#Documents per category statistics
		category_id = 0
		if self.verbose:
			print('Caculating docs per cat meta-features...')
		for category in self.cor.categories():
			#print(str(i) + ' '+category)
			docs_num = len(self.cor.fileids(category))
			self.docs_per_category.append(docs_num)
			self.imbalance+=pow(docs_num-endc,2)
			category_id+=1
			vocabulary_size = 0
			this_cat_vocabulary = set([])
			files_count = 0
			#Counting vocabulary size per category
			if self.verbose:
				print('Counting vocabulary...')
			for doc in self.cor.fileids(category):
				if self.limit and files_count >= self.per_cat_limit:
					if self.verbose:
						print('Limit reached for category, sample taken')
					break
				try:
					doc_words = self.cor.words(doc)
					vocabulary = set(doc_words)
					#raw_text = self.cor.raw(doc)
				except:
					raw_document = open(self.path+doc, errors='ignore')
					raw_text = raw_document.read()
					doc_words = word_tokenize(raw_text)
					vocabulary = set(doc_words)
				finally:
					doc_vocabulary_size = len(vocabulary)
					vocabulary_size += doc_vocabulary_size
					document_vocabulary_size.append(doc_vocabulary_size)
					this_cat_vocabulary.update(vocabulary)
					raw_documents_path.append(self.path+doc)
					target_documents.append(category_id)
					files_count+=1
			
			category_vocabulary_size.append(vocabulary_size)
			category_vocabulary.append(this_cat_vocabulary)
			
		self.imbalance = np.sqrt(self.imbalance/self.number_of_categories)

		nobs, self.minmax_dpc, self.avg_dpc, self.sd_dpc, self.skw_dpc, self.kur_dpc = describe(self.docs_per_category)
		self.sd_dpc=np.sqrt(self.sd_dpc)
		self.ratio_avg_sd_dpc = self.avg_dpc/self.sd_dpc
		self.entropy_dpc = entropy(self.docs_per_category, base = 2)
		
		if self.verbose:
			print('Using BoW...')
		if self.verbose:
			print('pca meta-features...')
		#BoW statistics / traditional meta-features
		n_docs = len(document_vocabulary_size)
		count_vect = CountVectorizer(input = 'filename', decode_error = 'ignore', max_features = 25000)
		X_train_counts = count_vect.fit_transform(raw_documents_path)
		tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
		X_train_tf = tf_transformer.transform(X_train_counts)
		svd = TruncatedSVD(n_components=100)
		svd.fit(X_train_tf)
		self.pcap_bow = svd.explained_variance_[0]
		nobs, self.pcaminmax_bow, self.pcaavg_bow, self.pcavar_bow, self.pcaskw_bow, self.pcakur_bow = describe(svd.components_[0])
		for p in range(len(svd.explained_variance_)):
			if p < 10:
				self.pcac_bow += svd.singular_values_[p]
			self.pca_singular_sum_bow += svd.singular_values_[p]
			self.pca_explained_ratio_bow += svd.explained_variance_ratio_[p]
			self.pca_explained_var_bow += svd.explained_variance_[p]
		self.pcac_bow = self.pcac_bow / self.pca_singular_sum_bow
		
		self.zero_pct = X_train_tf.getnnz()
		self.zero_pct = 1 - (self.zero_pct / (30000*n_docs))
		
		if self.verbose:
			print('Landmarking meta-features...')
		
		X_train, X_test, y_train, y_test = train_test_split(X_train_tf, target_documents, test_size=0.3, random_state=50)
		knn_clsf = KNeighborsClassifier(n_neighbors=1)
		knn_clsf.fit(X_train, y_train)
		predictions = knn_clsf.predict(X_test)
		self.onenn = accuracy_score(y_test, predictions)
		predictions = []
		
		tree_clsf = tree.DecisionTreeClassifier()
		tree_clsf.fit(X_train, y_train)
		predictions = tree_clsf.predict(X_test)
		self.tree = accuracy_score(y_test, predictions)
		predictions = []
		
		lin_clsf = LogisticRegression(solver='saga', n_jobs=2)
		lin_clsf.fit(X_train, y_train)
		predictions = lin_clsf.predict(X_test)
		self.lin = accuracy_score(y_test, predictions)
		predictions = []
		
		nb_clsf = MultinomialNB()
		nb_clsf.fit(X_train, y_train)
		predictions = nb_clsf.predict(X_test)
		self.nb = accuracy_score(y_test, predictions)
		
		if self.verbose:
			print('Caculating words per doc meta-features...')
		#Words per document statistics
		i = 0

		files_count = 0
		for doc in self.cor.fileids():
			if self.limit and files_count >= self.docs_limit:
				if self.verbose:
					print('Limit reached, sample taken')
				break
			try:
				doc_words = self.cor.words(doc)
				words_num = len(doc_words)
			except:
				raw_document = open(self.path+doc, errors='ignore')
				raw_text = raw_document.read()
				raw_document.close()
				doc_words = word_tokenize(raw_text)
				words_num = len(doc_words)
			finally:
				for word in doc_words:
					all_words.append(word)
					self.word_length += len(word)
				files_count += 1

			self.words_per_doc.append(words_num)
			i+=1

		self.word_length = self.word_length / len(all_words)
		nobs, self.minmax_wpd, self.avg_wpd, self.sd_wpd, self.skw_wpd, self.kur_wpd = describe(self.words_per_doc)
		self.sd_wpd=np.sqrt(self.sd_wpd)
		self.ratio_avg_sd_wpd = self.avg_wpd/self.sd_wpd
		self.entropy_wpd = entropy(self.words_per_doc, base = 2)
		
		if self.verbose:
			print('POS tagging meta-features...')
		
		#POS tagging statistics
		tagged_words = nltk.pos_tag(all_words, tagset='universal')
		tag_counts = Counter(tag for word,tag in tagged_words)
		total_tagged = sum(tag_counts.values())
		tag_pct = dict((word, float(count)/total_tagged) for word,count in tag_counts.items())
		self.adj_pct = tag_pct['ADJ']
		self.adp_pct = tag_pct['ADP']
		self.adv_pct = tag_pct['ADV']
		self.conj_pct = tag_pct['CONJ']
		self.det_pct = tag_pct['DET']
		self.noun_pct = tag_pct['NOUN']
		self.num_pct = tag_pct['NUM']
		self.prt_pct = tag_pct['PRT']
		self.pron_pct = tag_pct['PRON']
		self.verb_pct = tag_pct['VERB']
		self.mark_pct = tag_pct['.']
		self.other_pct = tag_pct['X']
		
		if self.verbose:
			print('Harness meta-features...')
		
		#SEM
		words_list_freq = []
		words_distribution = nltk.FreqDist(all_words)
		#print(words_distribution.most_common(10))
		for sample in words_distribution:
			words_list_freq.append(words_distribution[sample])
		words_list_freq.sort(reverse=True)
		terms_num = len(words_list_freq) #Vocabulary size
		terms_total = sum(words_list_freq)
		zipf_total = 0
		for i in range(1,terms_num+1):
			zipf_total+=(1/i)
		term_probability = []
		zipf_distribution = []
		
		for i in range(terms_num):
			term_probability.append(words_list_freq[i]/terms_total)
			zipf_distribution.append((1/(i+1))/zipf_total)
			self.SEM += term_probability[i] * np.log2(term_probability[i]/zipf_distribution[i])
		
		#UVB
		#print("Number of docs???: "+str(n_docs))
		for doc in document_vocabulary_size:
			self.UVB += pow((doc - terms_num)/n_docs,2)
		self.UVB = self.UVB/n_docs
		self.UVB = math.sqrt(self.UVB)
		
		#SVB
		for category in category_vocabulary_size:
			self.SVB += pow((category - terms_num)/n_docs,2)
		self.SVB = self.SVB/self.number_of_categories
		self.SVB = math.sqrt(self.SVB)
		
#		mrhs = []
#		mrhi = []
#		mrhj = []
		#MRH_J
		for cati in range(self.number_of_categories-1):
			for catj in range(cati+1, self.number_of_categories):
				valor = len(category_vocabulary[cati].intersection(category_vocabulary[catj])) / len(category_vocabulary[cati] | category_vocabulary[catj])
#				mrhs.append(valor)
#				mrhi.append(cati)
#				mrhj.append(catj)
				self.MRH_J += valor
#		print('more related')
#		top5 = sorted(range(len(mrhs)), key=lambda i: mrhs[i])[-5:]
#		for i in top5:
#			print(str(mrhi[i]) + ' ' + str(mrhj[i]) + ' with: ' + str(mrhs[i]))

		#VL
		self.VL = terms_num / len(self.words_per_doc)
		
		#VDR
		self.VDR = np.log2(self.VL) / np.log2(self.avg_wpd)
		
		#Vocabulary stats
		self.size_voc, self.minmax_voc, self.avg_voc, self.sd_voc, self.skw_voc, self.kur_voc = describe(words_list_freq)
		self.sd_voc = np.sqrt(self.sd_voc)
		self.ratio_avg_sd_voc = self.avg_voc/self.sd_voc
		self.entropy_voc = entropy(words_list_freq, base = 2)
		


		if self.verbose:
			print('Textstats...')

		#Readability
		try:
			#complete_text = self.cor.raw()
			complete_text = ''
			raw_texts = []
			files_count = 0
			for doc in self.cor.fileids():
				if self.limit and files_count >= self.docs_limit:
					if self.verbose:
						print('Textstats: Limit reached, sample taken')
					break
				raw_document = open(self.path+doc, errors='ignore')
				raw_texts.append(raw_document.read())
				files_count += 1
			complete_text = ''.join(raw_texts)
		except Exception as e:
			print(e)
					
		self.reading_ease = textstat.flesch_reading_ease(complete_text)
		self.smog_index = textstat.smog_index(complete_text)
		self.kincaid_grade = textstat.flesch_kincaid_grade(complete_text)
		self.coleman_liau_index = textstat.coleman_liau_index(complete_text)
		self.readability_index = textstat.automated_readability_index(complete_text)
		self.dale_chall_score = textstat.dale_chall_readability_score(complete_text)
		self.difficult_words = textstat.difficult_words(complete_text)
		self.linsear_formula = textstat.linsear_write_formula(complete_text)
		self.guning_fog = textstat.gunning_fog(complete_text)
		string_grade = textstat.text_standard(complete_text)
		matches = re.findall(r'\d+',string_grade)
		self.school_grade = matches[0]

	def get_meta_features(self):
		feats = []

		feats.append(self.number_of_documents)
		feats.append(self.number_of_categories)
		feats.append(self.word_length)
		feats.append(self.minmax_dpc[0])
		feats.append(self.minmax_dpc[1])
		feats.append(self.avg_dpc)
		feats.append(self.sd_dpc)
		feats.append(self.skw_dpc)
		feats.append(self.kur_dpc)
		feats.append(self.ratio_avg_sd_dpc)
		feats.append(self.entropy_dpc)
		feats.append(self.minmax_wpd[0])
		feats.append(self.minmax_wpd[1])
		feats.append(self.avg_wpd)
		feats.append(self.sd_wpd)
		feats.append(self.skw_wpd)
		feats.append(self.kur_wpd)
		feats.append(self.ratio_avg_sd_wpd)
		feats.append(self.entropy_wpd)
		feats.append(self.imbalance)
		feats.append(self.SEM)
		feats.append(self.UVB)
		feats.append(self.SVB)
		feats.append(self.MRH_J)
		feats.append(self.VL)
		feats.append(self.VDR)
		feats.append(self.size_voc)
		#feats.append(self.minmax_voc[0])
		feats.append(self.minmax_voc[1])
		feats.append(self.avg_voc)
		feats.append(self.sd_voc)
		feats.append(self.skw_voc)
		feats.append(self.kur_voc)
		feats.append(self.ratio_avg_sd_voc)
		feats.append(self.entropy_voc)
		feats.append(self.pcac_bow)
		feats.append(self.pca_singular_sum_bow)
		feats.append(self.pca_explained_ratio_bow)
		feats.append(self.pca_explained_var_bow)
		feats.append(self.pcap_bow)
		feats.append(self.pcaminmax_bow[0])
		feats.append(self.pcaminmax_bow[1])
		feats.append(self.pcaavg_bow)
		feats.append(self.pcavar_bow)
		feats.append(self.pcaskw_bow)
		feats.append(self.pcakur_bow)
		feats.append(self.onenn)
		feats.append(self.tree)
		feats.append(self.lin)
		feats.append(self.nb)
		feats.append(self.zero_pct)
		feats.append(self.adj_pct)
		feats.append(self.adp_pct)
		feats.append(self.adv_pct)
		feats.append(self.conj_pct)
		feats.append(self.det_pct)
		feats.append(self.noun_pct)
		feats.append(self.num_pct)
		feats.append(self.prt_pct)
		feats.append(self.pron_pct)
		feats.append(self.verb_pct)
		feats.append(self.mark_pct)
		feats.append(self.other_pct)
		feats.append(self.reading_ease)
		feats.append(self.smog_index)
		feats.append(self.kincaid_grade)
		feats.append(self.coleman_liau_index)
		feats.append(self.readability_index)
		feats.append(self.dale_chall_score)
		feats.append(self.difficult_words)
		feats.append(self.linsear_formula)
		feats.append(self.guning_fog)
		feats.append(self.school_grade)
		feats = np.array(feats)
		feats = feats.astype(np.float32)
		feats = np.nan_to_num(feats)
		#print(feats)
		return feats
		
