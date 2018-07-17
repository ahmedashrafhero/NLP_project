# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 17:17:19 2018

@author: merna
"""

from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import string
import pickle
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
nb = MultinomialNB()
nb.fit(X_train, y_train)
preds = nb.predict(X_test)
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))
print('\n')
print(accuracy_score(y_test,preds))

