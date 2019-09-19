#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:42:43 2019

@author: bruno
"""

import os
import cv2
from tqdm import tqdm
import random
import numpy as np
import keras


class ImageProcessing():
    def __init__(self):
        self.images = []
        self.classe = []


    def loadImages(self, diretory, sizeImg, channels = 0):
        """  Load Imges  """
        direImages = os.listdir(diretory)
        direImages = random.sample(direImages, len(direImages))
        for i in tqdm(direImages):
            if i != '.DS_Store':
                name = i.split('_')[0]
                img = cv2.imread(diretory+i, channels)
                img = cv2.resize(img,(sizeImg,sizeImg))
                img = img.astype('float32')/255
                if name == 'normal':
                    self.images.append(img)
                    self.classe.append(0)
                else:
                    self.images.append(img)
                    self.classe.append(1)
        images =  np.array(self.images)
        classe = np.array(self.classe)

        images = images.reshape(images.shape[0], sizeImg,sizeImg,1)
        classe = keras.utils.to_categorical(classe,2)

        return images,classe

