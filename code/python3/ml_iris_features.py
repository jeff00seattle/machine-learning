#!/usr/bin/env plabel_setthon3
# -*- coding: utf-8 -*-

# Show desision serface.

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def visualize_sepal_data():
    iris = load_iris()
    feature_set = iris.data[:, :2]  # we onllabel_set take the first two features.
    label_set = iris.target
    # plt.scatter(feature_set[:, 0], feature_set[:, 1], c=label_set, cmap=plt.cm.brg)
    for target in set(iris.target):
        x = [feature_set[i, 0] for i in range(len(iris.target)) if iris.target[i] == target]
        y = [feature_set[i, 1] for i in range(len(iris.target)) if iris.target[i] == target]
        plt.scatter(x, y, color=['red', 'blue', 'green'][target], label=iris.target_names[target])
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Sepal Width & Length')
    plt.legend(iris.target_names, loc='lower right')
    plt.show()


def visualize_petal_data():
    iris = load_iris()
    feature_set = iris.data[:, 2:]  # we onllabel_set take the last two features.
    label_set = iris.target
    # plt.scatter(feature_set[:, 0], feature_set[:, 1], c=label_set, cmap=plt.cm.brg)
    for target in set(iris.target):
        x = [feature_set[i, 0] for i in range(len(iris.target)) if iris.target[i] == target]
        y = [feature_set[i, 1] for i in range(len(iris.target)) if iris.target[i] == target]
        plt.scatter(x, y, color=['red', 'blue', 'green'][target], label=iris.target_names[target])
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.title('Petal Width & Length')
    plt.legend(iris.target_names, loc='lower right')
    plt.show()


if __name__ == '__main__':
    visualize_sepal_data()
    visualize_petal_data()