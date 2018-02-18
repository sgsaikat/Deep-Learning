# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 00:42:26 2017

@author: Saika
"""
import tensorflow as tf
import tensorflow.contrib.learn as learn
from sklearn.datasets import load_wine
wine = load_wine()

X = wine['data']
y_int = wine['target']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

##############################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_int, test_size=0.33, random_state=42)

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

classifier = learn.DNNClassifier(hidden_units=[20,20,10], n_classes=3, optimizer='Adam', \
                                 activation_fn=tf.nn.relu, feature_columns=feature_columns)
classifier.fit(X_train, y_train, steps=100, batch_size=3)
predictions = list(classifier.predict(X_test, as_iterable=True))

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

##############################################################

from keras.utils.np_utils import to_categorical
y = to_categorical(y_int)

from keras.layers import Dense
from keras.models import Sequential

def build_model():
    classifier = Sequential()
    classifier.add(Dense(units=32, activation='relu', input_dim=X.shape[1]))
    classifier.add(Dense(units=16, activation='relu'))
    classifier.add(Dense(units=16, activation='relu'))
    classifier.add(Dense(units=y.shape[1], activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

from keras.wrappers.scikit_learn import KerasClassifier
estimator = KerasClassifier(build_fn=build_model, batch_size=5, epochs=100, verbose=1)

from sklearn.model_selection import cross_val_score, KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(estimator, X, y, cv=kfold)
print("\n" + "Mean Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("\n" + "Max Accuracy: %.2f%%" % (results.max()*100))
print("\n" + "Min Accuracy: %.2f%%" % (results.min()*100))

##############################################################
classifier = Sequential()
classifier.add(Dense(units=32, activation='relu', input_dim=X.shape[1]))
classifier.add(Dense(units=16, activation='relu'))
classifier.add(Dense(units=y.shape[1], activation='sigmoid'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
predictions = classifier.predict(X_test)
