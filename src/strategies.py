from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import csv
import numpy as np

class RepresentationSearch:
	
	def __init__(self, X, exclude = ''):
		self.exclude = exclude
		self.X_test = np.array(X)
		self.X_test = np.nan_to_num(self.X_test)
		data = []
		with open('../tasks.csv','r') as f_corpora:
			f_reader = csv.reader(f_corpora, delimiter = ',')
			for row in f_reader:
				data.append(row)
		corpus_names = []
		X_train = []
		y_train = []
		y_perfs = []
		y_ranks = []
		for row in data[1:]:
			if(exclude == row[0]):
				#self.X_test = list(map(float,row[1:73]))
				continue
			corpus_names.append(row[0])
			y_train.append(row[84])
			X_train.append(list(map(float,row[1:73])))
		
		performances = []
		ranks = []
		for name in corpus_names:
			with open('../metadata/'+name+'.csv','r') as f_performance:
				i = 0
				f_reader = csv.reader(f_performance, delimiter = ',')
				for row in f_reader:
					performances.append(row[0])
					ranks.append(row[1])
					i+=1
					if(i>=60):
						break
			y_perfs.append(performances)
			y_ranks.append(ranks)
			performances = []
			ranks = []
			
		self.y_rankings = np.matrix.transpose(np.array(y_ranks))
		self.y_performances = np.matrix.transpose(np.array(y_perfs))
		self.y_train = np.array(y_train)
		self.X = np.array(X_train)
		
	def nearest_predict(self):
		nearest = KNeighborsClassifier(n_neighbors = 1)
		nearest.fit(self.X, self.y_train)
		return nearest.predict(self.X_test.reshape(1,-1))

	def classif_predict(self):
		classif = RandomForestClassifier(n_estimators = 100)
		classif.fit(self.X, self.y_train)
		return classif.predict(self.X_test.reshape(1,-1))
	
	#A regressor for each pipeline		
	def regressor_predict(self):
		reg = RandomForestRegressor(n_estimators = 100)
		predictions = []
		for p in self.y_performances:
			reg.fit(self.X, p)
			prediction = reg.predict(self.X_test.reshape(1,-1))
			predictions.append(float(prediction[0]))
		return np.argmax(predictions)
		
	def rank_predict(self):
		reg = RandomForestRegressor(n_estimators = 100)
		predictions = []
		for p in self.y_rankings:
			reg.fit(self.X, p)
			prediction = reg.predict(self.X_test.reshape(1,-1))
			predictions.append(np.floor(float(prediction[0])))
		return np.argmin(predictions)
		
	def top5(self):
		reg = RandomForestRegressor(n_estimators = 100)
		predictions = []
		for p in self.y_rankings:
			reg.fit(self.X, p)
			prediction = reg.predict(self.X_test.reshape(1,-1))
			predictions.append(np.floor(float(prediction[0])))
		return np.argpartition(predictions, 5)[:5]
	
	def all_r(self):
		return np.array(list(range(1,56)))
