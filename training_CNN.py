#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:23:06 2019

@author: bruno
"""


import image_processing


dire_train = 'data/train/'
dire_test = 'data/test/'
dire_val = 'data/val/'


sizeImg = 200

images = image_processing.ImageProcessing()

x_train, y_train = images.loadImages(dire_train,sizeImg)
x_test, y_test = images.loadImages(dire_test,sizeImg)


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from keras.optimizers import Adam
EPOCHS = 20
INIT_LR = 1e-3
BS = 40
opt = Adam(lr=INIT_LR, decay=INIT_LR / 100)

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size = 3, activation = 'relu', input_shape=(sizeImg,sizeImg,1)))
model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
model.add(Conv2D(16, kernel_size = 3, activation = 'relu'))
model.add(Flatten())

model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer = opt, loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

filepath='model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

history = model.fit_generator(
	aug.flow(x_train, y_train, batch_size=BS),
	validation_data=(x_test, y_test),
	steps_per_epoch=len(x_train) // BS,
	epochs=EPOCHS, verbose=1, callbacks = callbacks_list)