# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 00:11:48 2017

@author: Saika
"""
import tensorflow as tf
import tensorflow.contrib.learn as learn

from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()

X = iris['data']
y = iris['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

classifier = learn.DNNClassifier(hidden_units=[20,20,20], n_classes=3, optimizer='Adam', \
                                 activation_fn=tf.nn.sigmoid, feature_columns=feature_columns)
classifier.fit(X_train, y_train, steps=1900, batch_size=5)
iris_predictions = list(classifier.predict(X_test, as_iterable=True))

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test, iris_predictions))
print('\n')
print(classification_report(y_test, iris_predictions))