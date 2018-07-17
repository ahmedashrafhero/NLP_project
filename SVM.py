# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 17:04:19 2018

@author: merna
"""

from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn import cross_validation
#from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
yelp = pd.read_csv('movie-pang02.csv')
X = yelp['text']
y = yelp['class']
print(X)
print(X.shape)
bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
X = bow_transformer.transform(X)
 #لغاية هنا كدا دا كان جزء الpreprocessing
X_train, X_test, y_train, y_test =cross_validation.train_test_split(X, y, test_size=0.3, random_state=101)
clf = LinearSVC(random_state=0,max_iter=2000)
clf.fit(X_train, y_train)
print(clf.coef_)
print(clf.intercept_)
pred=clf.predict(X_test)
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))
print('\n')
print(accuracy_score(y_test,pred))