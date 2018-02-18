# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 02:30:01 2017

@author: Saika
"""
import matplotlib.pyplot as plt
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#plt.imshow(X_train[59999],cmap='Greys')

#from keras.utils.np_utils import to_categorical
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization

classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), input_shape=(28,28,2), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#classifier.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu'))
#classifier.add(MaxPooling2D(pool_size=(2,2)))
#classifier.add(Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), activation='relu'))
#classifier.add(MaxPooling2D(pool_size=(2,2)))
#classifier.add(Conv2D(filters=8, kernel_size=(3,3), strides=(2,2), activation='relu'))
#classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=784, activation='relu'))
classifier.add(Dense(units=10, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit_generator((X_train, y_train), steps_per_epoch=80, epochs=2, validation_data=y_train, validation_steps=20)