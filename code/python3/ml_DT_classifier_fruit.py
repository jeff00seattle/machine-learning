#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Classification: Fruit

# Follow a recipe for supervised learning (a technique to create a classifier from examples)
# and code it up.



# Examples
# Weight Texture Label
# 150g   Bumpy   Orange
# 170g   Bumpy   Orange
# 140g   Smooth  Apple
# 130g   Smooth  Apple

# Training Data
# Features: [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]  # Input to classifier
feature_set = [[140, 1], [130, 1], [150, 0], [170, 0]]  # scikit-learn uses real-valued feature_set
feature_names = ["Weight (grams)", "Texture"]  # Input to classifier
# Labels: ["apple", "apple", "orange", "orange"]  # Desired output
label_set = [0, 0, 1, 1]
label_names = ["apple", "orange"]

# Train Classifer
from sklearn import tree
clf = tree.DecisionTreeClassifier()  # Decision Tree classifier
clf = clf.fit(feature_set, label_set)  # Find patterns in data

# Make Predictions
print(clf.predict([[160, 0]]))
# Output: 0-apple, 1-orange
# Correct output is: 1-orange

# Visualize
# from scikit decision tree tutorial: http://scikit-learn.org/stable/modules/tree.html
from sklearn.externals.six import StringIO
import pydotplus
import os

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=feature_names,
                     class_names=label_names,
                     filled=True, rounded=True,
                     impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
os.system("rm ./ml_images/ml_decision_tree_fruit.pdf")
graph.write_pdf("./ml_images/ml_decision_tree_fruit.pdf")
os.system("open ./ml_images/ml_decision_tree_fruit.pdf")
