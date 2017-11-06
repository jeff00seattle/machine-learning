#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Let's Write a Pipeline

# How to test a model and determine accuracy

# Partition data into 2 sets, train and test

# import a dataset
from sklearn import datasets

iris = datasets.load_iris()

# Can think of classifier as a function f(x) = y
feature_set = iris.data  # features
label_set = iris.target  # labels

# partition into training and testing sets
from sklearn.model_selection import train_test_split

# test_size=0.5 -> split in half
x_train, x_test, y_train, y_test = train_test_split(feature_set, label_set, test_size=0.5)

# Nearest Neighbors Classifier (KNN)
# http://scikit-learn.org/stable/modules/neighbors.html
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)

# predict
predictions = clf.predict(x_test)
print(predictions)

# test
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))