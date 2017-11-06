#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://blog.socialcops.com/engineering/machine-learning-python/

import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, model_selection, tree, preprocessing, metrics
import sklearn.ensemble as ske
import tensorflow as tf
import tensorflow.contrib.learn as skflow


titanic_df = pd.read_csv('../../../datasets/titanic_passengers_wikipedia.csv', na_values=['NA'])
print(titanic_df.count())

print("Survival Mean: {0}".format(titanic_df['survived'].mean()))
print(titanic_df.groupby('pclass').mean())

class_sex_grouping = titanic_df.groupby(['pclass','sex']).mean()
print(class_sex_grouping)
class_sex_grouping_survived = class_sex_grouping['survived']
print(class_sex_grouping_survived)

ax = class_sex_grouping_survived.plot(kind='bar', title ="class sex survival", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("pclass", fontsize=12)
ax.set_ylabel("survival", fontsize=12)
plt.show()

group_by_age = pd.cut(titanic_df["age"], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping_survived = age_grouping['survived']
print(age_grouping_survived)
ax = age_grouping_survived.plot(kind='bar', title ="age survival", figsize=(15, 10), legend=True, fontsize=12)
ax.set_ylabel("survived", fontsize=12)
plt.show()

# Preparing The Data

print(titanic_df.count())
titanic_df = titanic_df.drop(['body','cabin','boat'], axis=1)

titanic_df["home.dest"] = titanic_df["home.dest"].fillna("NA")
titanic_df = titanic_df.dropna()
print(titanic_df.count())