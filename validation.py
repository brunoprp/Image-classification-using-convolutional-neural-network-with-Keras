#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:09:01 2019

@author: bruno
"""


from sklearn.metrics import accuracy_score, classification_report 
import numpy as np
from keras.models import load_model
import image_processing #import the image processing script

dire_val = 'data/val/' #validation diretory
sizeImg = 200 #image dimension
images = image_processing.ImageProcessing() #Instantiating an object from class (ImageProcessing)
x_val, y_val = images.loadImages(dire_val,sizeImg)


model_path = 'model.h5'
model = load_model(model_path, compile=False)


yhat_probs = model.predict(x_val, verbose=0)
yhat_classes = np.argmax(yhat_probs,axis=1)
y = np.argmax(y_val,axis=1)
accuracy = accuracy_score(y , yhat_classes)
print(classification_report(y_val.argmax(axis=1), yhat_probs.argmax(axis=1)))
print(accuracy)