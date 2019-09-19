#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:23:06 2019

@author: bruno
"""


import image_processing #import the image processing script

dire_train = 'data/train/' #trainig diretory
dire_test = 'data/test/' #test diretory

##########Reading the test and training images!############
sizeImg = 200 #image dimension
images = image_processing.ImageProcessing() #Instantiating an object from class (ImageProcessing)

x_train, y_train = images.loadImages(dire_train,sizeImg) #Trainig image processing 
x_test, y_test = images.loadImages(dire_test,sizeImg) #Test imge processing

######## Creating the model of convolutional neural network ################

"""Importing the Keras library for model building"""
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

"""Setting Network Hyperparameters"""
EPOCHS = 20 #Epocha number
INIT_LR = 1e-3 #Learning rate
BS = 40 # batch size, amount of images that will pass through the network in each season
opt = Adam(lr=INIT_LR, decay=INIT_LR / 100) #optimizer

model = Sequential() # Sequential model keras
#add model layers
"""Convolutional layers"""
model.add(Conv2D(64, kernel_size = 3, activation = 'relu', input_shape=(sizeImg,sizeImg,1))) #
model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
model.add(Conv2D(16, kernel_size = 3, activation = 'relu'))
model.add(Flatten())
"""Dense layers"""
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))# the last layer must have the same number of classes
model.compile(optimizer = opt, loss='categorical_crossentropy',metrics=['accuracy']) #Compiling the model by defining the optimizer the loss function and the evaluation metric
model.summary()#Print CNN architecture

#########Training CNN#######

"""Save model treined"""
filepath='model.h5'#name model save
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')#salve best model
callbacks_list = [checkpoint]

"""here a keras technique is used to improve training accuracy, the images are molded for the network to learn better"""
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

"""CNN trained by passing training and test data jutamnete with hyperparameters"""
history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1, callbacks = callbacks_list)