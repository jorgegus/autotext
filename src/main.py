from autotext import Autotext
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

root = '../datasets/'
train_sets = ['20_newsgroups/']
test_sets = ['20_newsgroups/']
skip_sets = ['20news-18828']

	
autotext = Autotext( strategy = 'classif', limit_memory = True)

#try:
autotext.train(root+train_sets[0], skip = skip_sets[0])
predictions = autotext.predict(root+test_sets[0])
'''
except Exception as e:
	predictions = []
	print('ERROR: ')
	print(e)
'''
a = accuracy_score(predictions,autotext.y_test)
f = f1_score(predictions,autotext.y_test,average='macro')
p = precision_score(predictions,autotext.y_test,average='macro')
r = recall_score(predictions,autotext.y_test,average='macro')
with open('../results/performance.csv', 'w+') as f_results:
	rwriter = csv.writer(f_results, delimiter=',')
	rwriter.writerow(['Dataset', 'Accuracy', 'F1', 'Precission', 'Recall'])
	rwriter.writerow([test_sets[0], a, f, p, r])
print(train_sets[0]+ ': '+str(a))
