from __future__ import print_function, unicode_literals

try:
    import numpy
except ImportError:
    pass

import tempfile
import os
from collections import defaultdict

from six import integer_types

from nltk import compat
from nltk.data import gzip_open_unicode
from nltk.util import OrderedDict
from nltk.probability import DictionaryProbDist

from nltk.classify.api import ClassifierI
from nltk.classify.util import CutoffChecker, accuracy, log_likelihood
from nltk.classify.megam import call_megam, write_megam_file, parse_megam_weights
from nltk.classify.tadm import call_tadm, write_tadm_file, parse_tadm_weights

__docformat__ = 'epytext en'


#import
import re
import string
import nltk
import ssl
import pandas as pd
import numpy as np 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.classify.maxent import MaxentFeatureEncodingI

#read in training data 
training_data = open('training_set.csv','r')
#next(training_data)

#first pass at feature selection, each review, a feature for every unigram in review
#tokenize reviews based on whitespace, remove punctuation, convert all letters to lowercase
allReviews = []
allWords = []

for line in training_data:
    line = re.sub('[^A-Za-z0-9]+', ' ', line)

    review = line.lower()
    allReviews.append(review)


#print(allReviews)

ps = PorterStemmer()

positiveReviews = []
negativeReviews = []
trainPositive = []
trainNegative =[]



for review in allReviews:
    words = review.split()
    for word in words:
        word = re.sub('[^A-Za-z0-9]+', ' ', word)
        sentiment = words[-1]
       

   
    if "0" in sentiment:
       negativeReviews[review] = review
    if "1" in sentiment:
        '''
        positiveReviews.append(review)
        trainPositive.append(1)
        '''
    
    allWords.append(words)

testReviews = []
test_data = open('testing_set.csv','r')
for line in test_data:
    line = re.sub('[^A-Za-z0-9]+', ' ', line)

    review = line.lower()
    testReviews.append(review)


train  = [(trainPositive, 'positive'),(trainNegative, 'negative')]
test = [testReviews]
#, negativeReviews, 'negative']


def test_maxent(algorithms) :
    classifiers = {}
    for algorithm in nltk.classify.MaxentClassifier.ALGORITHMS:
        if algorithm.lower() == 'megam':
            try: nltk.classify.megam.config_megam()
            except: raise #continure
        classifiers[algorithm] = nltk.MaxentClassifier.train(train, algorithm, trace = 0, max_iter = 1000)
    print (' '*11 +''.join(['       test[%s]   ' % i for i in range(len(test))]))
    print(' '*11+'       p(x)   p(y)'*len(test))
    print( '-'*(11+15*len(test)))
    for algorithm, classifier in classifiers.items():
        print('%11s' % algorithm)
        for featureset in test:
            pdist = classifier.prob_classify(featureset)
            print('%8.2f%6.2f' % (pdist.prob('x'), pdist.prob('y')))
        
test_maxent(nltk.classify.MaxentClassifier.ALGORITHMS)

#MaxentFeatureEncodingI.train(cls, pos_train_toks)
