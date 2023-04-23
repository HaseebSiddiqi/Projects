# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:09:37 2023

@author: Haseeb Siddiqi 301229958 Ensemble Learning 
"""


import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Load the data

path = "C:/Users/hasee/OneDrive/Desktop/Semester 4/Supervised learning/Ensemble Learning"
filename = 'pima-indians-diabetes.csv'
fullpath = os.path.join(path,filename)
df_haseeb = pd.read_csv(fullpath, sep=',')

df_haseeb.columns = ['preg', 'plas', 'press', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

print(df_haseeb.info())
print(df_haseeb.isnull().sum())
print(df_haseeb.describe())
print(df_haseeb['class'].value_counts())

#Pre-process and prepare the data for machine learning 

transformer_haseeb = StandardScaler()
numeric_cols = ['preg', 'plas', 'press', 'skin', 'test', 'mass', 'pedi', 'age']
df_haseeb[numeric_cols] = transformer_haseeb.fit_transform(df_haseeb[numeric_cols])

X_haseeb = df_haseeb.drop('class', axis=1)
y_haseeb = df_haseeb['class']

X_train_haseeb, X_test_haseeb, y_train_haseeb, y_test_haseeb = train_test_split(X_haseeb, y_haseeb, test_size=0.3, random_state=58)
X_train_haseeb[numeric_cols] = transformer_haseeb.transform(X_train_haseeb[numeric_cols])
X_test_haseeb[numeric_cols] = transformer_haseeb.transform(X_test_haseeb[numeric_cols])

#Exercise 1: Hard Voting 

log_haseeb = LogisticRegression(max_iter=1400)
log_haseeb.fit(X_train_haseeb, y_train_haseeb)
rfc_haseeb = RandomForestClassifier()
rfc_haseeb.fit(X_train_haseeb, y_train_haseeb)
svc_haseeb = SVC()
svc_haseeb.fit(X_train_haseeb, y_train_haseeb)
dtc_haseeb = DecisionTreeClassifier(criterion="entropy", max_depth=42)
dtc_haseeb.fit(X_train_haseeb, y_train_haseeb)
etc_haseeb = ExtraTreesClassifier()
etc_haseeb.fit(X_train_haseeb, y_train_haseeb)


voting_haseeb = VotingClassifier(
    estimators=[
        ('log', log_haseeb),
        ('rfc', rfc_haseeb),
        ('svc', svc_haseeb),
        ('dtc', dtc_haseeb),
        ('etc', etc_haseeb)
    ],
    voting='hard'
)

voting_haseeb.fit(X_train_haseeb, y_train_haseeb)
predictions_haseeb = voting_haseeb.predict(X_test_haseeb.iloc[:3, :])


classifiers_haseeb = [log_haseeb, rfc_haseeb, svc_haseeb, dtc_haseeb, etc_haseeb, voting_haseeb]

for clf in classifiers_haseeb:
    print(f"{clf.__class__.__name__}:")
    for i in range(3):
        X_instance = X_test_haseeb.iloc[i, :]
        y_pred = clf.predict(X_instance.values.reshape(1, -1))[0]
        y_true = y_test_haseeb.iloc[i]
        print(f"\tInstance {i+1}: Predicted={y_pred}, Actual={y_true}")
        
#Exercise #2: Soft voting

svc_haseeb = SVC(probability=True)
svc_haseeb.fit(X_train_haseeb, y_train_haseeb)

svc_haseeb.fit(X_train_haseeb, y_train_haseeb)

voting_haseeb_soft = VotingClassifier(
    estimators=[
        ('log', log_haseeb),
        ('rfc', rfc_haseeb),
        ('svc', svc_haseeb),
        ('dtc', dtc_haseeb),
        ('etc', etc_haseeb)
    ],
    voting='soft'
)

voting_haseeb_soft.fit(X_train_haseeb, y_train_haseeb)
predictions_haseeb_soft = voting_haseeb_soft.predict(X_test_haseeb.iloc[:3, :])

classifiers_haseeb_soft = [log_haseeb, rfc_haseeb, svc_haseeb, dtc_haseeb, etc_haseeb, voting_haseeb_soft]

for clf in classifiers_haseeb_soft:
    print(f"{clf.__class__.__name__}:")
    for i in range(3):
        X_instance = X_test_haseeb.iloc[i, :]
        y_pred = clf.predict(X_instance.values.reshape(1, -1))[0]
        y_true = y_test_haseeb.iloc[i]
        print(f"\tInstance {i+1}: Predicted={y_pred}, Actual={y_true}")
        
#Exercise #3: Random forests & Extra Trees 

pipeline1_haseeb = Pipeline([
    ('transformer_haseeb', transformer_haseeb),
    ('etc_haseeb', etc_haseeb)
])

pipeline2_haseeb = Pipeline([
    ('transformer_haseeb', transformer_haseeb),
    ('dtc_haseeb', dtc_haseeb)
])

pipeline1_haseeb.fit(X_haseeb, y_haseeb)
pipeline2_haseeb.fit(X_haseeb, y_haseeb)

# Shuffle the data
X_shuffled, y_shuffled = shuffle(X_haseeb, y_haseeb, random_state=42)

scores_pipeline1 = cross_val_score(pipeline1_haseeb, X_shuffled, y_shuffled, cv=10)
scores_pipeline2 = cross_val_score(pipeline2_haseeb, X_shuffled, y_shuffled, cv=10)

print(f"Mean score for Pipeline #1: {scores_pipeline1.mean()}")
print(f"Mean score for Pipeline #2: {scores_pipeline2.mean()}")

pipelines = [pipeline1_haseeb, pipeline2_haseeb]
pipeline_names = ['Pipeline #1', 'Pipeline #2']

for i in range(len(pipelines)):
    y_pred = pipelines[i].predict(X_test_haseeb)
    confusion = confusion_matrix(y_test_haseeb, y_pred)
    precision = precision_score(y_test_haseeb, y_pred, average='weighted')
    recall = recall_score(y_test_haseeb, y_pred, average='weighted')
    accuracy = accuracy_score(y_test_haseeb, y_pred)

    print(f"\nResults for {pipeline_names[i]}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion}")
   
#Exercise 4: Extra Trees and Grid search

n_estimators_58 = np.arange(10, 3000, 20)
max_depth_58 = np.arange(1, 1000, 2)

param_dist_58 = {'etc_haseeb__n_estimators': n_estimators_58,
                 'etc_haseeb__max_depth': max_depth_58}


grid_search_58 = RandomizedSearchCV(estimator=pipeline1_haseeb, param_distributions=param_dist_58, n_iter=50, cv=10, random_state=42, n_jobs=-1)

grid_search_58.fit(X_haseeb, y_haseeb)

print(f"Best Parameters: {grid_search_58.best_params_}")
print(f"Accuracy Score: {grid_search_58.best_score_}")

y_pred_58 = grid_search_58.best_estimator_.predict(X_test_haseeb)

confusion_58 = confusion_matrix(y_test_haseeb, y_pred_58)
precision_58 = precision_score(y_test_haseeb, y_pred_58, average='weighted')
recall_58 = recall_score(y_test_haseeb, y_pred_58, average='weighted')
accuracy_58 = accuracy_score(y_test_haseeb, y_pred_58)

print(f"Precision: {precision_58}")
print(f"Recall: {recall_58}")
print(f"Accuracy: {accuracy_58}")
