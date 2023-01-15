import pandas as pd
import os
import matplotlib.pyplot as plt 


# Loading .csv data into Panadas dataframe
path = "C:/Users/hasee/OneDrive/Desktop/Comment spam detector Naive bayes classifier"
filename = 'Youtube01-Psy.csv'
fullpath = os.path.join(path,filename)
group1_data = pd.read_csv(fullpath, sep=',')


# Trim columns to necessary feature and class columns: CONTENT & CLASS & toLower
group1_data_trim = group1_data.drop(columns=["COMMENT_ID", "AUTHOR", "DATE"], axis=1);
group1_data_trim = group1_data_trim.rename(columns={'CONTENT': 'content', 'CLASS': 'class'})


# Data Exploration
group1_data_trim.head(3)
group1_data_trim.shape
group1_data_trim.columns.values
group1_data_trim.dtypes
group1_data_trim.isnull().sum()

# Shuffling data
group1_data_trim = group1_data_trim.sample(frac = 1)
group1_data_trim.head(3)


# Splitting data for test and train
# (350, 2) 25%=88 for test  75%=262 for train
group1_data_train = group1_data_trim.iloc[:262,:]
group1_data_test = group1_data_trim.iloc[262:,:]
xTrain = group1_data_train.iloc[:,:1]
yTrain = group1_data_train.iloc[:,1:]
xTest = group1_data_test.iloc[:,:1]
yTest = group1_data_test.iloc[:,1:]


# Vectorize training data
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(xTrain['content'])

print("\nDimensions of training data:", train_tc.shape)


# Getting term frequencies from vectorized training data
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
type(train_tfidf)
print(train_tfidf.shape)
print(train_tfidf)


# Create and train a Multinomial Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
yTrain = yTrain.to_numpy().reshape(-1)
group1Classifier = MultinomialNB().fit(train_tfidf, yTrain)


# Cross validating the model based on training data 5-fold
from sklearn.model_selection import cross_val_score
scores = cross_val_score(group1Classifier, train_tfidf, yTrain, cv=5)
print("Training Accuracy Mean: ",scores.mean())


# Vectorize testing data
test_tc = count_vectorizer.transform(xTest.content)
type(test_tc)
print(test_tc)


# Getting term frequencies from vectorized testing data
test_tfidf = tfidf.transform(test_tc)
type(test_tfidf)
print(test_tfidf)


# Testing the model
predictions = group1Classifier.predict(test_tfidf)
print(predictions)


# Creating a dataframe to compare yPred with yActual
yActual = yTest.to_numpy().reshape(-1)
yCompare = pd.DataFrame(predictions, columns=["yPred"])
yCompare['yActual'] = yActual
print("Number of correct predictions based on testing data =",sum(a == b for a,b in zip(predictions, yActual)),"out of 88")


# Printing confusion matrix 
from sklearn.metrics import confusion_matrix
group1_confusion_matrix = confusion_matrix(yTest, predictions)
print(group1_confusion_matrix)
from sklearn import metrics
print('Accuracy based on yTest data', metrics.accuracy_score(yTest, predictions))

import seaborn as sns
#From https://www.jcchouinard.com/
ax = sns.heatmap(group1_confusion_matrix, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix based on Testing Data');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
## Display the visualization of the Confusion Matrix.
plt.show()


##############################################################################


# Testing the trained group1Classifier with our own inputs
group_content = [
    'I love South Korean food', #Ham
    'Psy is the coolest k-pop star', #Ham
    'This music is awesome and better than Beethoven & Mozart', #Ham
    'Can anyone help me with my homework?', #Ham
    'I need help this song is too catchy', #Ham
    'How is everyone feeling after listening to this song?', #Ham
    'What is the weather in Canada like today?', #Ham
    'Visit this mental health website if you need help: https://cmha.ca/', #Ham
    'Like and subscribe to my YouTube channel  https://www.youtube.com/channel/UCF7IcRT05226IXBo7eUMfzg', #Spam
    'Someone please venmo me cash or buy me giftcards so I can become a singer like Psy', #Spammish
    'If you like Psy, you will love this https://www.youtube.com/@BLACKPINK' #Spam
    ]
# 1 = Spam
# 0 = Ham
group_yActual = [0,0,0,0,0,0,0,0,1,1,1]


# Vectorize group input data
group_tc = count_vectorizer.transform(group_content)
type(group_tc)
print(group_tc)


# Getting term frequencies from vectorized testing data
group_tfidf = tfidf.transform(group_tc)
type(group_tfidf)
print(group_tfidf)


# Testing the model
group_predictions = group1Classifier.predict(group_tfidf)
print(group_predictions)


# Creating a dataframe to compare yPred with group_yActual
yGroupCompare = pd.DataFrame(group_predictions, columns=["Group_Y_Pred"])
yGroupCompare['Group_Y_Actual'] = group_yActual
print("Number of correct predictions =",sum(a == b for a,b in zip(group_predictions, group_yActual)),"out of 11")


# Printing confusion matrix 
group_content_confusion_matrix = confusion_matrix(group_yActual, group_predictions)
print(group_content_confusion_matrix)
from sklearn import metrics
print('Accuracy based on group input data', metrics.accuracy_score(group_yActual, group_predictions))
#From https://www.jcchouinard.com/
ax = sns.heatmap(group_content_confusion_matrix, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix based on group input data');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
## Display the visualization of the Confusion Matrix.
plt.show()