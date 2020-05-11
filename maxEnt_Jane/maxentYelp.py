  
#import the files 
import re 
import nltk
import ssl
import pandas as pd 
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import linear_model
#from sklearn.metrics import f1_Score

cv = CountVectorizer()
train_data_2 = pd.read_csv('training_set.csv')
test_data = pd.read_csv('testing_set.csv')


X_train = cv.fit_transform(train_data_2["Review"]).toarray()
y_train = train_data_2["Liked"].values
X_test = cv.transform(test_data["Review"]).toarray()
y_test = test_data["Liked"].values
print(X_train)


#(X, y) = (features matrix, labels)
maxent = linear_model.LogisticRegression(penalty = 'none', C=1.0)
maxent.fit(X_train, y_train)
#print (maxent.coef_)
#Matrix with shape (n classes, n features)

#predict vector with (integer) labels
y_predicted = maxent.predict(X_test)
test_accuracy = accuracy_score(y_test, y_predicted)
#print(test_accuracy)

# probability distribution over all possible classes
# Shape: (n_instances, n_classes)
y_probs = maxent.predict_proba(X_test)

#compute F-scores for all classes

#confusion matrix
cm = confusion_matrix(y_test, y_predicted)
#print("Confusion Matrix:\n", cm)
