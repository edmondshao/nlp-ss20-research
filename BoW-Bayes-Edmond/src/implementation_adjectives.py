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

# if the packages arent downloaded on your local run this
""" try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download() """

stop_words = list(set(stopwords.words('english')))
# JJ/R/S = adjective/comparitive/superlative
allowed_word_types = ["JJ","JJR","JJS"]
#open the data 
test_data = pd.read_csv('testing_set.csv')
train_data = pd.read_csv('training_set.csv')

#input tokenized list of words from review 
#return tokenized list without stop words 
def remove_stopword(data_input):
    cleaned_words = []
    for word in data_input: 
        if word not in stop_words: 
            cleaned_words.append(word)
    return (cleaned_words)

# input tokenized and pos tagged list 
# return list of just adjectives
def only_adjectives(data_input):
    cleaned_words = []
    for word, pos in data_input: 
        if pos in allowed_word_types:
            cleaned_words.append(word)
    return (cleaned_words)


#clean the training set for everything except for adjectives 
corpus = []

#lower case 
train_data["Review"] = train_data["Review"].apply(lambda x:x.lower())
#get rid of punctuation
train_data["Review"] = train_data["Review"].apply(lambda x:re.sub('[^a-zA-Z]',' ',x))
#split
train_data["Review"] = train_data["Review"].apply(lambda x:x.split())
#remove the stop words 
train_data["Review"] = train_data["Review"].apply(lambda x: remove_stopword(x))

#pos tag --> creates tuple (word,pos-tag)
train_data["Review"] = train_data["Review"] = train_data["Review"].apply(lambda x:nltk.pos_tag(x))
#Select only adjectives and remove pos-tags
train_data["Review"] = train_data["Review"].apply(lambda x:only_adjectives(x))



#create list of all adjectives from reviews in train_data
all_review_train = []
for row in train_data["Review"]:
    for word in row:
        all_review_train.append(word)


# BoW using count vectorizer 
cv = CountVectorizer()
cv.fit(all_review_train)

train_data_2 = pd.read_csv('training_set.csv')
test_data = pd.read_csv('testing_set.csv')

X_train = cv.transform(train_data_2["Review"]).toarray()
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
