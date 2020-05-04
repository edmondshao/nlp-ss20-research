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

cv = CountVectorizer()
train_data_2 = pd.read_csv('training_set.csv')
test_data = pd.read_csv('testing_set.csv')


X_train = cv.fit_transform(train_data_2["Review"]).toarray()
y_train = train_data_2["Liked"].values
X_test = cv.transform(test_data["Review"]).toarray()
y_test = test_data["Liked"].values


#train the model 
classifier = BernoulliNB(alpha=0.8)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
score1 = accuracy_score(y_test,y_pred)

score2 = precision_score(y_test,y_pred)
score3= recall_score(y_test,y_pred)
print("\n")
print("Accuracy is ",round(score1*100,2),"%")
print("Precision is ",round(score2,2))
print("Recall is ",round(score3,2))