from __future__ import absolute_import, division, print_function, unicode_literals

# Maths Functions
import numpy as np
import math
from random import randint

# Neural Network
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU, Lambda
from keras import backend as K, optimizers
import h5py

# File manipulation
import os, os.path
import gc
import pandas as pd
import glob
from matplotlib import image, pyplot as plt

# Class Files
from ImageManipulation import Graph
from LabelConverter import FourArrayToArray, TTypeToBinary, ModifiedTTypeToBinary, ClassificationToBinary

# Paths and Variables
Print_Images = False                                                                            # Prints the first 20 images of a batch if true
Simple_Model = False                                                                            # Whether to use a simple or complex model
Method= 2                                                                                       # Which study images to load (1= GZ2, 2=NA10, 3= EFIGI)
size = 96                                                                                       # Size of images to open
DR = 16                                                                                         # The data release of the training images
TestDR = 16                                                                                     # The data release of the testing images
Version = 3.1                                                                                   # Model version
File_Type = "jpg"                                                                               # File type of test images
if(Simple_Model):
    Model_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Models\GalaxyClass_V" + str(Version) +"_DR" + str(DR) + str(size) + ".h5"          # Simple Model path
else:
    Model_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Models\GalaxyClass_FeatureV" + str(Version) +"_DR" + str(DR) + str(size) + ".h5"   # Complex model path

if(Method == 1):    # GZ2
    Image_Path = r"F:\samue\Pictures\GZ2_" + str(size) + "_DR" + str(DR)                                                                                                                                    # Where the GZ2 images are
    Map_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\GZ2_DataV1.txt"                                                                                                                  # Where the GZ2 labels are
    if(Simple_Model):
        Class_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Galaxy Zoo 2 Results\GZ_Class_Results_V" + str(Version) +"DR" + str(DR) + str(size) + "_DR" + str(TestDR) + ".txt"         # Where you want to save the probabilites this code produces of GZ2
    else:
        Class_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Galaxy Zoo 2 Results\GZ_Class_Results_FeatureV" + str(Version) +"DR" + str(DR) + str(size) + "_DR" + str(TestDR) + ".txt"
if(Method ==2):     # NA10
    Image_Path = r"F:\samue\Pictures\Galaxies_" + str(size) + "_Central"+ "_DR" + str(DR)
    Map_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\AbrahamDatav3.txt"
    if(Simple_Model):
        Class_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\N&A Results\NA_Class_Results_V" + str(Version) +"DR" + str(DR) + str(size) + "_DR" + str(TestDR) + ".txt"
    else:
        Class_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\N&A Results\NA_Class_Results_FeatureV" + str(Version) +"DR" + str(DR) + str(size) + "_DR" + str(TestDR) + ".txt"
if(Method ==3):     # EFIGI
    Image_Path = r"F:\samue\Pictures\EFIGI_" + str(size) + "_DR" + str(DR)
    Map_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\EFIGI_DataV1.txt"
    if(Simple_Model):
        Class_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\EFIGI Results\EFIGI_Class_Results_V" + str(Version) +"DR" + str(DR) + str(size) + "_DR" + str(TestDR) + ".txt"
    else:
        Class_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\EFIGI Results\EFIGI_Class_Results_FeatureV" + str(Version) +"DR" + str(DR) + str(size) + "_DR" + str(TestDR) + ".txt"

# Script variables
Counter = 0                                                                                     # Counts the number of files opened
Batch = 1                                                                                       # Prints the number of batches to the screen
Sum1 = 0                                                                                        # Counts the number of images loaded
Sum2 = 0                                                                                        # Counts the number of labels loaded
Num_Files = len(glob.glob(os.path.join(Image_Path, "*." + File_Type)))                          # Number of files in the directory
Txt_Array = np.empty((0, 8))                                                                                                                                                                                # This is the size of the probability text file, I think youll need to change this to 6

model = load_model(Model_Path)                                                                  # Loads model from directory                                                                                # Change this to load weights, add compile etc. Labels are loaded in sparse format
model.summary()                                                                                 # Gives an overview of the model

if(Method == 1):
    Map = pd.read_csv(Map_Path, delim_whitespace=True, header=None, names=["id", "z", "RA", "DEG", "class"])    # Opens a mapping file 
    Class = ClassificationToBinary(np.squeeze(np.array(Map.loc[:,['class']])))                                  # Creates an array of only classification types
    ID = np.squeeze(np.array(Map.loc[:,['id']]))                                                                # Creates an array of IDs
    Redshift = np.squeeze(np.array(Map.loc[:,['z']]))
if(Method == 2 or Method == 3):
    Map = pd.read_csv(Map_Path, delim_whitespace=True, header=None, names=["id", "z", "RA", "DEG", "TType"])
    Class = ModifiedTTypeToBinary(np.squeeze(np.array(Map.loc[:,['TType']])) + 7, False)                        # Converts TTypes to Binary
    ID = np.squeeze(np.array(Map.loc[:,['id']]))
    Redshift = np.squeeze(np.array(Map.loc[:,['z']]))

print("Starting testing")
while True:                                                                                                                                     # Continues to loop until the counter is greater the number of files in the directory 
    
    print("Performing tests on batch " + str(Batch))
    Batch += 1
    
    Batch_images = []                                                                                                
    Batch_Labels = []
    Batch_IDs = []                                                                                               
    for i in range(Counter, 13000+Counter):                                                                                                     # Opens 13000 images from image path
        Counter += 1                                                                                                                            # Counter measures the number of images loaded to limit the loop
        if(Counter > len(Class)): #or Counter > Num_Files ):                                                                                    # Breaks loop if counter reaches the number of files in directory or the number of labels in the mapping file  
            break
        
        Batch_Path = os.path.join(Image_Path, str(int(ID[i])) + "." + File_Type)
        if(os.path.exists(Batch_Path)):
            if(Redshift[i]>0.08):
                continue
            if(Class[i] ==5):
                continue
            Batch_images.append(image.imread(Batch_Path))                                                                                       # Opens file (if it exists) from image path with asset ID and file type
            Batch_Labels.append(Class[i])
            Batch_IDs.append(ID[i])                                                                                                             # Adds the images classification to a classification array
        else:
            continue
    print("Loaded " + str(len(Batch_images)) + " images and " + str(len(Batch_Labels)) + " labels.")                                
    Images = np.asarray(Batch_images)/255.0                                                                                                     # Normalises colour
    Labels = np.asarray(Batch_Labels)
    Asset_IDs= np.asarray(Batch_IDs)                                                                                                            # IDs for text file
    Sum1 += len(Batch_images)                                                                                                                  
    Sum2 += len(Batch_Labels)

    if(Print_Images):                                                                                                                           # Prints the first 20 images to the screen if true
        Graph(Images)

    score_1 = model.evaluate(Images, Labels, verbose= 1, use_multiprocessing= True)                                                             # Evaluates model with test data
    print('Test Loss', score_1[0])                                                                                                              # Loss score average of total data set
    print('Test Accuracy', score_1[1])                                                                                                          # Accuracy average of total data set                                                                               
    Txt_Array = np.append(Txt_Array, FourArrayToArray(Asset_IDs, Labels, model.predict_classes(Images), model.predict(Images)), axis=0)         # Adds all data from batch to an array used in saving to text file           

    if(Counter > len(Class)): #or Counter > Num_Files):                                                                                         # Breaks loop if counter reaches the number of files in directory or the number of labels in the mapping file  
        break

np.savetxt(Class_Path, Txt_Array, delimiter= "\t", newline="\n", fmt= "%i %i %i %1.3f %1.4f %1.4f %1.4f %1.4f")     # Writes probabilities to text file                                                             # fmt will need to be changed to accommodate change in text array size (6 long) 
           #,header= "Asset_ID    Class    Predicted_Class    Bias    S0_Prob    S0a_Prob    E_Prob    S_Prob")

print("Processed " + str(Sum1) +" images and " + str(Sum2) + " labels.")                                            # Prints total number of images loaded
print("Document saved")