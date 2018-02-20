# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 21:49:22 2018

@author: Jeet
"""

import pandas as pd
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold

housingDS = pd.read_csv('housing_data.csv', delim_whitespace=True, header=None)
housingDS = housingDS.values
x = housingDS[:,:-1]
y = housingDS[:,-1]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

def build_model():
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

seed = 12
numpy.random.seed(seed)
estimator = KerasRegressor(build_fn=build_model, epochs=50, batch_size=5)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, x, y, cv=kfold, verbose=1)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))