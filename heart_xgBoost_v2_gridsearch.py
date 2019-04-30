# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:49:54 2019

@author: fjehlik
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Select directory to import data
os.chdir(r"C:\Users\fjehlik\Documents\01 ML kaggle\01 heart disease UCI")

# Import and read Excel file data and transform to dataframe
df = pd.DataFrame()
file = 'heart.csv'
df = pd.read_csv(file, header=0)

# Establish features and target
feat = [
        'age' , \
        'sex' , \
        'cp' , \
        'trestbps' , \
        'chol' , \
        'fbs' , \
        'restecg' , \
        'thalach' , \
        'exang' , \
        'oldpeak' , \
        'slope' , \
        'ca' , \
        'thal']

targ = 'target'

# Establish the features and targets
X = df[feat]
y = df[targ]

# split data into train and test sets
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=16)

# Establish XGB classifier and the grid to search
clf = xgb.XGBClassifier()
grid = {'learning_rate':[0.01, 0.1, 0.15, 1], 'n_estimators': [25, 50, 100, 120, 140], 'max_depth':[1,2,3]}

# Block out the following code if you want to run the best parameters below
CV_clf = GridSearchCV(estimator=clf, param_grid=grid, cv=5, verbose=2)
CV_clf.fit(X_train, y_train)
print(CV_clf.best_params_)

# Pass the best estimators to the clf and refit using best parameters
clf_best = xgb.XGBClassifier(**CV_clf.best_params_)
clf=clf_best.fit(X_train, y_train)
clf.fit(X_train, y_train)

# Best values are {'learning_rate': 0.15, 'max_depth': 1, 'n_estimators': 50}
#clf = xgb.XGBClassifier(learning_rate = 0.15, max_depth=1, n_estimators=50)
#clf.fit(X_train, y_train)
# make predictions for test data
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))