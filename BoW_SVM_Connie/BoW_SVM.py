import numpy as np
import pandas as pd
import nltk
import sklearn
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
#nltk.download('stopwords')

# import the dataset
training_dataset = pd.read_csv('../answer-key/training_set.csv')
testing_dataset = pd.read_csv('../answer-key/testing_set.csv')

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
cv = CountVectorizer()

# # summarize
# print(cv.vocabulary_)

# #encode document
count_occurs_train = cv.fit_transform(training_corpus).toarray()
y_train = training_dataset.iloc[:, 2].values
# print(y_train)

# # training dataset -> used to fit the model, testing dataset -> used to perform the predictions

testing_corpus = []
for i in range(0, 300):
    review = re.sub('[^a-zA-Z]', ' ', testing_dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    testing_corpus.append(review)

count_occurs_test = cv.transform(testing_corpus).toarray()
y_test = testing_dataset.iloc[:, 2].values


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(count_occurs_train,y_train)

predictions_SVM = SVM.predict(count_occurs_test)

print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, y_test) * 100, '%')
print("SVM Precision Score -> ", round(precision_score(predictions_SVM, y_test), 2))
print("SVM Recall Score -> ", round(recall_score(predictions_SVM, y_test), 2))

## Approximate Results: 
# SVM Accuracy Score ->  75.66666666666667 %
# SVM Precision Score ->  0.7
# SVM Recall Score ->  0.83