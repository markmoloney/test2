# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

X_train = pd.read_csv("../01 - Data/unbalanced_X_train.csv")
#y_train = pd.read_csv('../01 - Data/unbalanced_y_train.csv', header = None, index_col = 0, squeeze = bool)
#X_train = X_train.drop('Unnamed: 0', axis=1)
y_train = pd.read_csv("../01 - Data/unbalanced_y_train.csv", header = None)
# -

import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

X_train.shape

y_train.shape

# +
scores_df = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'run_time'])

models = [GaussianNB(), RandomForestClassifier(n_estimators=10), 
            SGDClassifier(), DecisionTreeClassifier(), XGBClassifier()]
names = ["Naive Bayes", "Random Forest Classifier", "SGD Classifier", "Decision Tree", "XGBClassifier"]

#models = [GaussianNB(), SGDClassifier()]
#names = ['Naive Bayes', 'SGD Classifier']

for model, name in zip(models, names):
    temp_list = []
    print(name)
    start = time.time()
    
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=10)
    
    for score in ["accuracy", "precision", "recall", "f1", 'roc_auc']:
        if score == 'accuracy':
            mean_score = accuracy_score(y_train, y_train_pred)
        elif score == 'precision':
            mean_score = precision_score(y_train, y_train_pred)          
        elif score == 'recall':
            mean_score = recall_score(y_train, y_train_pred)
        elif score == 'f1':
            mean_score = f1_score(y_train, y_train_pred)
        elif score == 'auc':
            mean_score = auc_roc_score(y_train, y_train_pred)
        
        print('{} mean : {}'.format(score, mean_score))
        temp_list.append(mean_score)
    
    #doing it this way takes more time, because it runs cross_val_score 4 different times
    #for score in ["accuracy", "precision", "recall", "f1"]:
     #   mean_score = cross_val_score(model, X_train, y_train, scoring=score, cv=10).mean()
     #   print('{} mean score: {}'.format(score, mean_score))
    
    
    temp_list.append(time.time() - start)
    print("time to run: {}".format(time.time() - start))
    print("\n")
    scores_df.loc[name] = temp_list
# -

scores_df

scores_df

for i in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'run_time']:
    scores_df.plot.bar(y = i) 




