#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

#########################################################

from sklearn import svm
import pandas as pd
import numpy as np

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
#creating classifier
clf = svm.SVC(C=10000,kernel='rbf')

#fitting data
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

#getting predictions
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

#get accuracy
from sklearn.metrics import accuracy_score

print(accuracy_score(pred, labels_test))

print(sum(pred))
#pred = pd.DataFrame(pred,columns=['class'])
#print(pred[pred['class']==1].count())

#print(pred[10],pred[26],pred[50])
#accuracy for linear kernel: 98.4%
#training time: 187.527 s
#prediction time: 20.246 s

#accuracy for optimized rbf kernel (C=10000) = 89.2

              