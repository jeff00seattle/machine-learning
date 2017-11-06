#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_titanic_all = pd.read_csv('../../datasets/titanic_passengers_wikipedia.csv')
print(df_titanic_all.sample(5).to_string())

import seaborn as sns
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=df_titanic_all)
plt.show()

sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=df_titanic_all,
              palette={"male": "blue", "female": "red"},
              markers=["*", "o"], linestyles=["-", "--"]);
plt.show()

def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5).apply(lambda x: round(x, 1))
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df


def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df


def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df


def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df


def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked', 'Boat', 'Body', 'Home.Dest'], axis=1)


def reorder_columns(df):
    return df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Lname', 'NamePrefix']]


def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    df = reorder_columns(df)
    return df


df_titanic = transform_features(df_titanic_all)
print(df_titanic.head().to_string())

df_titanic = df_titanic.reindex(np.random.permutation(df_titanic.index)).reset_index(drop=True)
print(df_titanic.head().to_string())

df_train=df_titanic.sample(frac=0.70,random_state=200).reset_index(drop=True)
df_test=df_titanic.drop(df_train.index).reset_index(drop=True)

df_train['PassengerId'] = range(1, len(df_train) + 1)
df_test['PassengerId'] = range(len(df_train) + 1, len(df_train) + len(df_test) + 1)

cols = list(df_train.columns.values)
cols.pop(cols.index('PassengerId'))
df_train = df_train[['PassengerId']+cols]
df_test = df_test[['PassengerId']+cols]

df_train.head()


from sklearn import preprocessing


def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

df_train, df_test = encode_features(df_train, df_test)
df_train.head()

from sklearn.model_selection import train_test_split

X_all = df_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = df_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

print("Create Model: Fitting and Tuning an Algorithm")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier.
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], # The number of trees in the forest.
              'max_features': ['log2', 'sqrt','auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 5, 8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data.
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

print("Validate Model with KFold")

from sklearn.model_selection import KFold

def run_kfold(clf):
    kf = KFold(n_splits=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf.split(X_all):
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))

run_kfold(clf)

print("Make Predictions against Test Data")

df_test_predict = df_test.copy()

y_ids = df_test_predict['PassengerId']
df_test_predict = df_test_predict.drop(['PassengerId', 'Survived'], axis=1)


y_predictions = clf.predict(df_test_predict)

df_predictions = pd.DataFrame({ 'PassengerId' : y_ids, 'Survived': y_predictions })
print(df_predictions.head())

y_test = df_test['Survived']
accuracy = accuracy_score(y_test, y_predictions)
print(accuracy)

# Visualize
# from scikit decision tree tutorial: http://scikit-learn.org/stable/modules/tree.html
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
import os

feature_names = list(X_train.columns)
class_names = ['Not Survived', 'Survived']

tree_file_dir = "./../../../images/titanic/"
os.system("rm -fR " + tree_file_dir)
os.system("mdkir -p " + tree_file_dir)

i_tree = 0
for tree_in_forest in clf.estimators_:
    dot_data = StringIO()
    tree_file_name = tree_file_dir + "ml_tree_titanic_" + str(i_tree) + ".pdf"
    os.system("rm -f " + tree_file_name)
    tree.export_graphviz(tree_in_forest, out_file=dot_data, feature_names=feature_names, class_names=class_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(tree_file_name)
    i_tree = i_tree + 1
