import numpy as np
import pandas as pd
import nltk
import sklearn
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
#nltk.download('stopwords')

# import the dataset
training_dataset = pd.read_csv('../answer-key/training_set.csv')
testing_dataset = pd.read_csv('../answer-key/testing_set.csv')
dataset = pd.read_csv('../answer-key/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# preprocess dataset: remove stopwords, numeric, and stop characters
training_corpus = []
for i in range(0, 700):
    review = re.sub('[^a-zA-Z]', ' ', training_dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    training_corpus.append(review)

# create bag of words model (use vectorization technique to convert textual data to numerical format)
# create the transform (using count word occurrence), keyword will occur again and again, number of occurrences = importance of word
# more frequency means more importance
cv = CountVectorizer(max_features = 1500)

# # summarize
# #print(cv.vocabulary_)

# #encode document
# count_occurs_train = cv.fit_transform(training_corpus).toarray()
# y_train = training_dataset.iloc[:, 1].values
# #pd.DataFrame(count_occurs_train).to_csv('bow.csv')
# # X_train, X_test, y_train, y_test = train_test_split(count_occurs, y)


# #TODO: prepare training set and test set? but already separated soooo do the same for test set as done in training set
# # training dataset -> used to fit the model, testing dataset -> used to perform the predictions

# testing_corpus = []
# for i in range(0, 300):
#     review = re.sub('[^a-zA-Z]', ' ', testing_dataset['Review'][i])
#     review = review.lower()
#     review = review.split()
#     review = [word for word in review if not word in set(stopwords.words('english'))]
#     review = ' '.join(review)
#     testing_corpus.append(review)

# count_occurs_test = cv.transform(testing_corpus).toarray()
# y_test = testing_dataset.iloc[:, 1].values

corpus =[]
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

# use SVM for classification
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train,Y_train)

#FIXME: error here!!! try splitting using train_test_split instead
predictions_SVM = SVM.predict(X_test)

print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Y_test)*100,'%')
print("SVM Precision Score -> ",round(precision_score(predictions_SVM, Y_test),2))
print("SVM Recall Score -> ",round(recall_score(predictions_SVM, Y_test),2))

# use testing_set to calculate accuracy (repeat above processes with testing set as well)