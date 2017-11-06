#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Show desision serface.

import numpy as np

import matplotlib.pyplot as plt

# Classifier
from sklearn.tree import DecisionTreeClassifier


# Parameters
n_classes = 3
plot_colors = "bg"
plot_step = 1

import numpy

feature_data = numpy.loadtxt(
    '../../../datasets/titanic_passengers_wikipedia.csv',
    delimiter=',',
    skiprows=1,
    usecols=(2,3,4,5)
)

feature_names = ["Ticket class", "Sex", "Age", "# Siblings/Spouses", "# Parents/Children"]

label_set = numpy.loadtxt(
    './ml_datasets/titanic_passengers_dataset_min.csv',
    delimiter=',',
    skiprows=1,
    usecols=(1,)
)

target_names = ["survived", "not survived"]

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):

    # We only take the two corresponding features
    feature_set = feature_data[:, pair]

    # Train
    clf = DecisionTreeClassifier().fit(feature_set, label_set)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = feature_set[:, 0].min() - 1, feature_set[:, 0].max() + 1
    y_min, y_max = feature_set[:, 1].min() - 1, feature_set[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    plt.xlabel(feature_names[pair[0]])
    plt.ylabel(feature_names[pair[1]])
    plt.axis("tight")

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(label_set == i)
        plt.scatter(feature_set[idx, 0], feature_set[idx, 1], c=color, label=target_names[i],
                    cmap=plt.cm.Paired)

    plt.axis("tight")

plt.suptitle("Decision surface of a '{0}' using paired features".format("Decision Tree (DT)"))
plt.legend()
plt.show()