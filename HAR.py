# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:02:33 2018

@author: Jeet
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report

activity = pd.read_table('activity_labels.txt', sep=' ', header=None, names=('ID','Label'))
features = pd.read_table('features.txt', sep=' ', header=None, names=('ID','Sensor'))
subject = pd.read_table('train\subject_train.txt', sep=' ', header=None, names=('Subject_ID',))

X_train = pd.read_table(r"train\X_train.txt", sep='\s+', header=None, names=features['Sensor'])
y_train = pd.read_table(r"train\y_train.txt", sep='\s+', header=None, names=('Activity_ID',))
y_train= OneHotEncoder().fit_transform(y_train)

X_test = pd.read_table(r"test\X_test.txt", sep='\s+', header=None, names=features['Sensor'])
y_test = pd.read_table(r"test\y_test.txt", sep='\s+', header=None, names=('Activity_ID',))
y_test = OneHotEncoder().fit_transform(y_test)

#from keras.utils.np_utils import to_categorical
#temp = to_categorical(y_train)            #Issue -> 6 Classes but temp created with shape-7

input_dim = X_train.shape[1]
model = Sequential()
model.add(Dense(output_dim=input_dim, kernel_initializer='normal', activation='relu', input_shape=(input_dim,)))
model.add(Dense(output_dim=280, kernel_initializer='normal', activation='relu'))
model.add(Dense(output_dim=140, kernel_initializer='normal', activation='relu'))
model.add(Dense(output_dim=70, kernel_initializer='normal', activation='relu'))
model.add(Dense(output_dim=6, kernel_initializer='normal', activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=5, epochs=20)
predictions = model.predict(X_test)
predictions = (predictions > 0.5)
print(confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1)))
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))

#[[484   9   3   0   0   0]
# [ 10 455   6   0   0   0]
# [  5  28 387   0   0   0]
# [  0   1   0 451  39   0]
# [  1   0   0  48 483   0]
# [  0   0   0   0   0 537]]
#             precision    recall  f1-score   support
#
#          0       0.97      0.98      0.97       496
#          1       0.92      0.97      0.94       471
#          2       0.98      0.92      0.95       420
#          3       0.90      0.92      0.91       491
#          4       0.93      0.91      0.92       532
#          5       1.00      1.00      1.00       537
#
#avg / total       0.95      0.95      0.95      2947