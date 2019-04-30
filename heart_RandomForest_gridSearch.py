# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:48:13 2019

@author: fjehlik

*****************************REFERENCE*********************************
Kaggle heart predictor:
https://www.kaggle.com/ronitf/heart-disease-uci


This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. 
In particular, the Cleveland database is the only one that has been used by ML researchers to this date. 
The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.


This program utilizes random forest classifier approach to predicting the outcome


age: age in years
sex: (1 = male; 0 = female)
cp: chest pain type
trestbps: resting blood pressure (in mm Hg on admission to the hospital)
chol: serum cholestoral in mg/dl
fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg: resting electrocardiographic results
thalach: maximum heart rate achieved
exang: exercise induced angina (1 = yes; 0 = no)
oldpeak: ST depression induced by exercise relative to rest
slope: the slope of the peak exercise ST segment
ca: number of major vessels (0-3) colored by flourosopy
thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
target: 1 or 0

***********************************************************************

"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Select directory to import data
os.chdir(r"C:\Users\fjehlik\Documents\01 ML kaggle\01 heart disease UCI")

# Import and read Excel file data and transform to dataframe
df = pd.DataFrame()
file = 'heart.csv'
df = pd.read_csv(file, header=0)

# Establish X and y
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

X = df[feat]
y = df[targ]

# Establisg the training and testing datatests
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 72)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# Randomforest classifier modeling section
clf = RandomForestClassifier(random_state = 72)
param_grid = { 
    'max_features': ['auto'],
    'n_estimators' : [10,50,100], 
    'max_depth' : [1,2,5,8],
    'min_samples_split' : [5,8,10], 
    'min_samples_leaf' : [5,10,15],
    'criterion' :['gini', 'entropy']
}

CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5, verbose=2)
CV_clf.fit(X_train, y_train)
print(CV_clf.best_params_)

# Pass the best estimators to the model and refit using best parameters
clf_best = RandomForestClassifier(**CV_clf.best_params_)
model=clf_best.fit(X_train, y_train)

# Use the best trained model to predict the blind test data
y_pred = clf_best.predict(X_test) 
score_train = clf_best.score(X_train, y_train)
score_test = clf_best.score(X_test, y_test)

print("Score train: " +str(round(score_train,3)))
print("Score test: " +str(round(score_test,3)))

# Extract the feature importances using .feature_importances_ 
importances = clf_best.feature_importances_

# Display the nine most important X
indices = np.argsort(importances)[::-1]
columns = X_train.columns.values[indices[:len(feat)]]
values = importances[indices][:len(feat)]

# Plot Univariate Histograms
X.hist(figsize=(10,12))
plt.show()

# Plot the weight impportances
fig = plt.figure(figsize=(10,12))
plt.title("Normalized Weights for Most Predictive X", fontsize = 12)
plt.bar(np.arange(len(feat)), values, width = 0.6, align="center", color = '#00A000', \
      label = "Feature Weight")
plt.xticks(np.arange(len(feat)), columns, rotation='vertical')
plt.xlim((-0.5, 13.5))
plt.ylabel("Weight", fontsize = 12)
plt.xlabel("Feature", fontsize = 12)
plt.legend(loc = 'upper right')
plt.show()  

# Plot the predicted vs actual for each point in counting order
count = np.linspace(0,len(y_test),num=len(y_test))
fig = plt.figure(figsize=(8,5))
plt.plot(count, y_test, 'o')
plt.plot(count, y_pred, 'x')
plt.ylabel("Target: 0=safe, 1=failure", fontsize = 12)
plt.xlabel("Patient number", fontsize = 12)
plt.legend(loc = 'best')
plt.show()  
