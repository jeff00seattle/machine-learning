#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Classification: Iris: Large Data Set for Training

# Build one on a real dataset, add code to visualize it, and practice reading it - so you can see how it works under the
# hood.

# Use Iris flower data set: https://en.wikipedia.org/wiki/Iris_flower_data_set
# Identify type of flower based on measurements
# Dataset includes 3 species of Iris flowers: setosa, versicolor, virginica
# 4 features used to describe: length and width of sepal and petal
# 50 examples of each type for 150 total examples

# Goals
# 1-Import dataset
# 2-Train a classifier
# 3-Predict label for new flower
# 4-Visualize the tree

# scikit-learn datasets: http://scikit-learn.org/stable/datasets/
# already includes Iris dataset: load_iris

from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

feature_set = iris.data  # features
label_set = iris.target  # labels

print(iris.feature_names)  # metadata: names of the features
print(iris.target_names)   # metadata: names of the different types of flowers

# print iris.data  # features and examples themselves
print(iris.data[0])  # first flower
print(iris.target[0])  # contains the labels

# partition into training and testing sets
from sklearn.model_selection import train_test_split

# test_size=0.5 -> split in half
train_data, test_data, train_target, test_target = train_test_split(feature_set, label_set, test_size=0.5)

# create new classifier
clf = tree.DecisionTreeClassifier()
# train on training data
clf.fit(train_data, train_target)

# what we expect
print(test_target)

# predict
predictions = clf.predict(test_data)
print("Predictions")
print(predictions)

# test
from sklearn.metrics import accuracy_score
print("Accuracy Score")
print(accuracy_score(test_target, predictions))

# Visualize
# from scikit decision tree tutorial: http://scikit-learn.org/stable/modules/tree.html
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus

print(iris.feature_names)
print(iris.target_names)

# Visualize
from IPython.display import Image
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())