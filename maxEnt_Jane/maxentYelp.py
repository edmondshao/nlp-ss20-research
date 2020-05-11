  
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
import scipy.stats as ss
import seaborn as sns
#from sklearn.metrics import f1_Score

cv = CountVectorizer()
train_data_2 = pd.read_csv('training_set.csv')
test_data = pd.read_csv('testing_set.csv')


X_train = cv.fit_transform(train_data_2["Review"]).toarray()
y_train = train_data_2["Liked"].values
X_test = cv.transform(test_data["Review"]).toarray()
y_test = test_data["Liked"].values



#(X, y) = (features matrix, labels)
maxent = linear_model.LogisticRegression(penalty = 'l2', C=1.0)
maxent.fit(X_train, y_train)
#print (maxent.coef_)
#Matrix with shape (n classes, n features)

#predict vector with (integer) labels
y_predicted = maxent.predict(X_test)
test_accuracy = accuracy_score(y_test, y_predicted)


# probability distribution over all possible classes
# Shape: (n_instances, n_classes)
y_probs = maxent.predict_proba(X_test)
score2 = precision_score(y_test,y_predicted)
score3= recall_score(y_test,y_predicted)
print("\n")
print("Accuracy is ",round(test_accuracy*100,2),"%")
print("Precision is ",round(score2,2))
print("Recall is ",round(score3,2))
print("\n")

#confusion matrix
cm = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:\n", cm)

#final
business = pd.read_csv("final_business.csv")
reviews = pd.read_csv("cleanish_reviews.csv")
ids = pd.read_csv('list_buiss_id.txt', sep = " ", header = None)
ids.columns = ["id"]

columns = ["business_id", "business_name", "our_score", "stars"]
results = pd.DataFrame(columns = columns)

for i in range(len(ids)):
    id_curr = ids.iloc[i].id

    all_reviews = reviews[reviews["business_id"]== id_curr]

    X_test = cv.transform(all_reviews["text"]).toarray()

    y_pred = maxent.predict(X_test)

    y_pred = np.mean(y_pred) * 100

    curr_buis = business.loc[business["business_id"]== id_curr]

    curr_star = (curr_buis["stars"].values[0]/5)*100

    results = results.append({"business_id":id_curr, "business_nanme":curr_buis["name"].values[0],"our_score":y_pred, "stars":curr_star}, ignore_index = True)

    results.head()

    ss.pearsonr(results["our_score"], results["stars"])

    g = sns.lmplot(x="our_score", y ="stars", data = results)

    results.filter(items = ["our_score", 'stars']).corr()

