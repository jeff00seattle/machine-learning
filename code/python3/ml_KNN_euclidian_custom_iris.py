#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Let's Write a Pipeline

# How to test a model and determine accuracy

# Partition data into 2 sets, train and test

# import a dataset
from sklearn.datasets import load_iris

import numpy

class KEuclidNeighborsClassifier:

    def __init__( self, k_neighbors = 3 ):
        self.k_neighbors = k_neighbors

    def fit( self, x_train, y_train ):
        self.x = x_train
        self.y = y_train

    def predict( self, testFeatures ):
        
        def euclid_dist( testPoint , checkPoint ):
            distance = numpy.linalg.norm( testPoint - checkPoint )
            return distance

        def closest( testPoint ):
            distArray = numpy.array( [ ( euclid_dist( testPoint , self.x[i] ), self.y[i] ) for i in range(len(self.x)) ] , dtype = [('dist', float),('lab', int)] )
            distArray.sort(order = 'dist')
            majority  = {}
            for j in range(self.k_neighbors):
                if majority.get( distArray[j][1] ) == None:
                    majority[ distArray[j][1] ] = 0
                else:
                    majority[ distArray[j][1] ] += 1

            return max( majority , key = majority.get )

        if testFeatures.ndim == 1:    #only one point to predict
            return closest( testFeatures )
        else:
            # prediction = []
            # for testPoint in testFeatures:
            #     prediction.append( closest( testPoint ) )
            # return numpy.array(prediction)            
            prediction = numpy.array( [closest(point) for point in testFeatures] )
            return prediction

def CheckClassifier( classifier , x_test , y_test ):
    total = len(y_test)
    correct = 0
    for i in range(total):
        prediction = classifier.predict( x_test[i] )
        if prediction == y_test[i]:
            correct += 1

    return (correct/total)*100

if __name__ == '__main__':
    iris = load_iris()
    
    # Can think of classifier as a function f(x) = y
    feature_set = iris.data  # features
    label_set = iris.target  # labels
    
    # partition into training and testing sets
    from sklearn.model_selection import train_test_split

    # test_size=0.5 -> split in half
    x_train, x_test, y_train, y_test = train_test_split(feature_set, label_set, test_size=0.5)
    
    clf = KEuclidNeighborsClassifier(k_neighbors=5)
    clf.fit(x_train, y_train)
    
    # predict
    predictions = clf.predict(x_test)
    print(predictions)
    
    # test
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, predictions))

    print('Test features: ', x_test[0])
    print('Test Label: ', y_test[0])
    print('Prediction: ', clf.predict( numpy.array( [24,51,28,58] ) ) )
    print(CheckClassifier(clf , x_test , y_test))