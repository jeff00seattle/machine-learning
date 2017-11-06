#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.pyplot as plt
from input import Input
from iris_nn import IrisNN
import numpy as np
from sklearn.metrics import f1_score

np.set_printoptions(threshold=np.inf, precision=3, linewidth=1000, suppress=True)

# Read the input file
input_data = Input('iris-flower-edited.csv', 4)
# The NN architecture & info
iris_nn = IrisNN(5, 4, 3)

# Gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(iris_nn.j)

# Training
session = tf.Session()
session.run(tf.initialize_all_variables())
epochs = 2500
cost = []
iteration = []
for epoch in range(epochs):
    cost.append(session.run(iris_nn.j, feed_dict={iris_nn.X: input_data.xs, iris_nn.Y: input_data.ys}))
    iteration.append(epoch)
    session.run(optimizer, feed_dict={iris_nn.X: input_data.xs, iris_nn.Y: input_data.ys})

y_pred = np.transpose(session.run(iris_nn.H, feed_dict={iris_nn.X: input_data.xs_test})).argmax(1)
y_true = np.transpose(input_data.ys_test).argmax(1)
f1 = f1_score(y_true, y_pred, average=None)

print('================================================================')
print('True: {}'.format(y_true))
print('Pred: {}'.format(y_pred))
print('F1 score: {!r}'.format(f1))
print('================================================================')

plt.ion()
fig, ax = plt.subplots(1, 1)
plt.plot(iteration, cost)
fig.show()
plt.draw()
plt.waitforbuttonpress()
