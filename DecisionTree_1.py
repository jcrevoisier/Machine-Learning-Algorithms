#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:52:57 2019

@author: jerome
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import graphviz

# Data importation
print("Data import ...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
headers = ["Class", "LW","LD","RW","RD"]
df = pd.read_csv(url, names=headers)

print("Data imported :\n")
print(df.describe())
print("\n") 
print(df[0:5])
print("... \n") 

# Train / Test split
print("Train / Test spliting ... ")
array = df.values
X = np.array(array[:,1:5])
Y = np.array(array[:,0])
m = X.shape[0]
n = X.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 7)
print("Train / Test splitting done \n")

# Training
print("Gini Training ... ")
clf = tree.DecisionTreeClassifier(criterion = "gini", random_state = 7 ,max_depth=5, min_samples_leaf=5)
clf.fit(X_train,Y_train)
print("Gini training done \n")

# Prediction
Y_pred = clf.predict(X_test)
print("Confusion Matrix :\n")
print(confusion_matrix(Y_test, Y_pred, labels=['L','B','R']))
print("\n")
print "Accuracy score :" , accuracy_score(Y_test, Y_pred)*100,"% \n"
print "Classification report:\n\n", classification_report(Y_test, Y_pred)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("iris") 