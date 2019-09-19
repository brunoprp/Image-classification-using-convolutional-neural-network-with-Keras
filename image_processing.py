#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:42:43 2019

@author: bruno
"""

import os #Read directories
import cv2 #Read the pictures
from tqdm import tqdm #Graphically show the progress of the 'FOR'
import random #Randomize the images
import numpy as np #Array work
import keras #Format the arrays

############### Creating a class to load and process images!######################

class ImageProcessing():
    def __init__(self):  #creating builder
        self.images = []
        self.classe = []


    def loadImages(self, diretory, sizeImg, channels = 0):#receive the directory the images the size they will be read and number of channels
        """  Load Imges  """
        direImages = os.listdir(diretory)#list everything in the directory
        direImages = random.sample(direImages, len(direImages))#Randomize images within directory
        for i in tqdm(direImages):#go through the directory
            if i != '.DS_Store':#ignore not image file inside directory
                name = i.split('_')[0]#capture image name (Normal or Pneumonia)
                img = cv2.imread(diretory+i, channels)#Load image
                img = cv2.resize(img,(sizeImg,sizeImg))# resize images
                img = img.astype('float32')/255 #Normalize image values
                #If the image name is normal the class is 0 if not the class is 1
                if name == 'normal': 
                    self.images.append(img)
                    self.classe.append(0)
                else:
                    self.images.append(img)
                    self.classe.append(1)
        images =  np.array(self.images)#Array numpy
        classe = np.array(self.classe)#Array numpy

        images = images.reshape(images.shape[0], sizeImg,sizeImg,1)#CNN input format (x,sizeImge,sizeImge,channels)
        classe = keras.utils.to_categorical(classe,2)#CNN input format class

        return images,classe #ruturn images and class

