# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:26:42 2017

@author: Saika
"""

import numpy as np
import pandas as pd
import keras
import seaborn as sns

from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()

#X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
x = iris['data']
y_int = iris['target']

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#x = sc.fit_transform(x)

from keras.utils.np_utils import to_categorical
y = to_categorical(y_int)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

def build_model():
    classifier = Sequential()
    classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu', input_dim=4))
    #classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=3, kernel_initializer='uniform', activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

from keras.wrappers.scikit_learn import KerasClassifier
estimator = KerasClassifier(build_fn=build_model, epochs=300, batch_size=5, verbose=1)

from sklearn.model_selection import cross_val_score, KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(estimator, x, y, cv=kfold)
print("\n" + "Mean Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("\n" + "Max Accuracy: %.2f%%" % (results.max()*100))
print("\n" + "Min Accuracy: %.2f%%" % (results.min()*100))