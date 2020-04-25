from __future__ import absolute_import, division, print_function, unicode_literals

# Maths Functions
import numpy as np
import math
from matplotlib import pyplot as plt
from random import randint

# Neural Network
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from keras import backend as K, optimizers, initializers, regularizers
import h5py

lrelu=lambda x: tf.keras.activations.relu(x, alpha=0.01)

# File manipulation
import os, os.path

# Class Files
from ImageManipulation import Load_Images, Graph

# Variables and Paths
Simple_Model = False
size = 96
DR = 16
Version = 3.2
Num_Epochs = 8
if(Simple_Model):
    Model_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Models\GalaxyClass_V" + str(Version) +"_DR" + str(DR) + str(size) + ".h5"
else:
    Model_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Models\GalaxyClass_FeatureV" + str(Version) +"_DR" + str(DR) + str(size) + ".h5"

# Loads images from computer into array using Load_Images, and then rotates all images in array increasing the
# array size by 4 times. The rotated images have the same TType as the original image, meaning the TType array 
# (label array) is also increased by 4.
# The images are split into 2 subsets, training images and test images which are then used to train and test the 
# machine later in the code.

data, data_labels, data_valid, data_labels_valid, Class_Weights = Load_Images(r'F:\samue\Pictures\Galaxies_' + str(size) + "_DR" + str(DR), 
                                                                              r'F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\AbrahamDatav2.1.txt',
                                                                              r"F:\samue\Pictures\EFIGI_"+ str(size)+ "_DR" + str(DR),
                                                                              r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\EFIGI_DataV1.txt",
                                                                              "jpg")                                                                            
# The input shape of the images for the first convolution layer. A normal image of n x n size is usually represented
# as a n x n x 3 matrix as the image is split into its RGB values.
#input_shape = data[0].shape

# The layers used in the model. A sequential model means the next layer connects to the previous layer, whereas a functional
# model means the next layer doesnt have to connect to the previous layer.
# Sequential - [] -> [] -> []
# 
# Functional - [] -> [] -> []
#              [] --↗            
model = Sequential()
if(Simple_Model):                                                                                               # Experimental CNN 
    model.add(Conv2D(32, kernel_size=(5, 5), padding = 'same', kernel_initializer="Orthogonal", bias_initializer= initializers.Constant(0.1), input_shape=(size,size,3)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, kernel_size=(3, 3), padding = 'same', kernel_initializer="Orthogonal", bias_initializer= initializers.Constant(0.1)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, kernel_size=(3,3), kernel_initializer="Orthogonal", bias_initializer= initializers.Constant(0.1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=(3,3), kernel_initializer="Orthogonal", bias_initializer= initializers.Constant(0.1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=(3,3), kernel_initializer="Orthogonal", bias_initializer= initializers.Constant(0.1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, kernel_size=(3,3), kernel_initializer="Orthogonal", bias_initializer= initializers.Constant(0.1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(128, kernel_size=(3,3), kernel_initializer="Orthogonal", bias_initializer= initializers.Constant(0.1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, kernel_size=(3,3), kernel_initializer="Orthogonal", bias_initializer= initializers.Constant(0.1), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(4608, kernel_initializer="Orthogonal", kernel_regularizer= regularizers.l2(0.0001), bias_initializer=initializers.Constant(0.01)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))
    model.add(Dense(4608, kernel_initializer="Orthogonal", kernel_regularizer= regularizers.l2(0.0001), bias_initializer=initializers.Constant(0.01)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))
    model.add(Dense(5, activation='softmax', kernel_initializer="Orthogonal", kernel_regularizer= regularizers.l2(0.0001), bias_initializer=initializers.Constant(0.01)))
else:                                                                                                           # Convoluted CNN
    model.add(Conv2D(32, kernel_size=(5,5), padding='same', kernel_initializer="random_normal", bias_initializer= initializers.Constant(0.1), input_shape = (size,size,3)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, kernel_size=(4,4), padding='same', kernel_initializer="random_normal", bias_initializer= initializers.Constant(0.1)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', kernel_initializer="random_normal", bias_initializer= initializers.Constant(0.1)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', kernel_initializer="random_normal", bias_initializer= initializers.Constant(0.1)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(4608, kernel_initializer="random_normal", kernel_regularizer= regularizers.l2(0.0001), bias_initializer=initializers.Constant(0.01)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.20))
    model.add(Dense(4608, kernel_initializer="random_normal", kernel_regularizer= regularizers.l2(0.0001), bias_initializer=initializers.Constant(0.01)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.20))
    model.add(Dense(5, activation="softmax", kernel_initializer="random_normal", kernel_regularizer= regularizers.l2(0.0001), bias_initializer=initializers.Constant(0.01), use_bias=False))
model.summary()

model.compile(optimizer= 'adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])                                                                                    # Hypothesis function of the model
model.fit(data, data_labels, epochs=Num_Epochs, validation_data=(data_valid, data_labels_valid), shuffle= True, use_multiprocessing= True) #, class_weight = Class_Weights)         # Fits data across model, validated against the validation data
model.save(Model_Path)                                                                                                                                                              # Saves model to a directory
print("Saved to drive") 