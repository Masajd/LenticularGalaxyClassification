# Mathmatical Functions
import numpy as np
import math
from random import randint
import random

# File and Image manipulation
import pandas as pd
import glob
from skimage.transform import resize
from matplotlib import image, pyplot as plt
from scipy import ndimage
from PIL import Image, ImageDraw
import os, os.path

# Nerual Network
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# Class files
from LabelConverter import TTypeToBinary, ModifiedTTypeToBinary, ClassificationToTType

# Rotates and crops a given image (original_image) from (0 + theta_delta) to theta_max by iterations of 1. 
# Example: Given a theta_max=90, theta_delta=0 and an image, this constructor would produce an array of 90 images. 
# The images are cropped to be the same size as the original image.
def image_rotator(original_image, original_image_labels):

    images = []
    images_labels = []
    #Num_Rots = 4

    for j in range(len(original_image)):                                              # Goes through all the original images, and creates 4 rotated images at 90,180,270 degress of each original image

        if(original_image_labels[j] == 4.0):                                            # If Label is a spiral then only 2 rotations to reduce skew in dataset
            Num_Rots=2
            Spiral = True
        else:
            Num_Rots=4
            Spiral = False

        for i in range(Num_Rots):

            if(Spiral):
                rot_image = ndimage.rotate(original_image[j],(randint(0,4)*90))     # Randomly rotates spiral galaxy by 0,90,180 or 270
            else:
                rot_image = ndimage.rotate(original_image[j],(i*90))                # Roates other galaxy types by 0,90,180,270
            #rot_image = ndimage.rotate(original_image[j],(i*90))
            crop_image = Crop_Square_Image(rot_image, original_image[j])            # Crops rotated image to be the same as the original image and then adds cropped image to array (necessary for images not rotated at 90 degrees)

            if(randint(0,1)==0):
                if(randint(0,1)==0):
                    images.append(crop_image)                                       # Adds rotated image to array
                else:
                    images.append(np.flipud(crop_image))                            # Flips the image along horizontal axis
            else:
                if(randint(0,1)==0):
                    images.append(np.fliplr(crop_image))                            # Flips the image along the vertical axis
                else:   
                    images.append(np.flipud(np.fliplr(crop_image)))                 # Flips the image along the veritcal and horizontal axis

            images_labels.append(original_image_labels[j])                          # Adds Galaxy TType to galaxy label array
    
    Combined = list(zip(images, images_labels))
    random.shuffle(Combined)
    Ran_Images, Ran_Labels = zip(*Combined)                                         # Randomises images to stop neural network from learning 'clumpiness' of data
    return Ran_Images, Ran_Labels

# Crops a rotated image (rot_image) to be the same size as the original SQUARE unrotated image (original_image).
# This constructor only works with SQUARE images as delta_xy calculates the height of an assumed square image.
def Crop_Square_Image(rot_image, original_image):
    
    # Finds the difference in height between the rotated image and the original image
    delta_xy = 0.5*(np.size(rot_image,0) - np.size(original_image,0)) 

    # Crops the image depending on whether its axises are even or odd in length (as crop has to be with integer values)
    if (np.size(rot_image,0)%2==0) & (np.size(rot_image,1)%2==0):

        image_crop = rot_image[int(delta_xy):(np.size(rot_image,0) - int(delta_xy)), int(delta_xy):(np.size(rot_image,0) - int(delta_xy))]  

    if (np.size(rot_image,0)%2!=0) & (np.size(rot_image,1)%2!=0):

        delta_xy_up = math.ceil(delta_xy)
        delta_xy_down = math.floor(delta_xy)
        image_crop = rot_image[delta_xy_up:(np.size(rot_image,0) - delta_xy_down), delta_xy_up:(np.size(rot_image,0) - delta_xy_down)]  

    return image_crop

# Resizes images to 64x64x3 images without any anti-aliasing
def Resize(Image):
    
    Resized_Images =[]
    for i in range(len(Image)):
        Resized_Image = resize(Image[i],(64,64), anti_aliasing=False)
        Resized_Images.append(Resized_Image)

    return Resized_Images

# Plots the first 20 images in the training images array into a 10 x 10 inch picture.
def Graph(Images, Limit):

    plt.figure(figsize=(10,10))
    for i in range(Limit):
        plt.subplot(6,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Images[i])
    plt.show()

# Randomises the data and corresponding labels to stop machine from learning any clumping in the dataset
def Randomiser(Images, Labels):
    
    Randomised_Images = []
    Randomised_Labels = []
    New_Images = Images
    New_Labels = Labels

    for i in range(len(Images)):

        New_len = len(New_Images)                                   # Gets length of image database
        Ran = randint(0,New_len-1)                                  # Gets a random number within the length of the database (starting from 0)
        
        Randomised_Images.append(New_Images[Ran])                   # Adds randomly picked image and label to another array
        Randomised_Labels.append(New_Labels[Ran])
        
        New_Images = Images[0:Ran] + Images[(Ran+1):New_len]        # Removes the randomly picked image and label from the original array
        New_Labels = Labels[0:Ran] + Labels[(Ran+1):New_len]        # Process repeats until all images are removed from the original array

    return Randomised_Images, Randomised_Labels

#Loads a large quantity of images from a path and puts them in an array.
def Load_Images(Train_Image_Path, Train_Label_Path, Val_Image_Path, Val_Label_Path, FileType):
    
    print("Opening Training data set.")
    Train_Class = pd.read_csv(Train_Label_Path, delim_whitespace=True,header=None, names=["ID", "z", "URL", "TType"])   # Opens the text file that contains the galaxy's TTypes
    TType = ModifiedTTypeToBinary(np.squeeze(np.array(Train_Class.loc[:,['TType']])) + 7, True)                                                      # Squeeze used to removed inner arrays (Label.loc produces a [[],[]] array which may cause problems further down the line)
    Train_Redshift = np.squeeze(np.array(Train_Class.loc[:,['z']]))                                                     # Redshift of galaxies
    Train_ID = np.squeeze(np.array(Train_Class.loc[:,['ID']]))                                                          # IDs of galaxies

    Train_File_Images = []
    Train_File_Labels = []
    for i in range(len(TType)):
        if(Train_Redshift[i]>0.08 or TType[i] == 5):                                                                                # If the redshift of a galaxy is greater than 0.08, it skips this galaxy
            continue
        else:
            Train_File_Images.append(image.imread(os.path.join(Train_Image_Path, str(int(Train_ID[i])) + "." + FileType)))          # Opens all images from Data_Path
            Train_File_Labels.append(TType[i])                                                                                      # Adds label of galaxy to label array
    print("Loaded " + str(len(Train_File_Images)) + " images and " + str(len(Train_File_Labels)) + " labels.")
    
    Summer(Train_File_Labels)                                                                          
    Rotate_Images, Rotate_Labels = image_rotator(Train_File_Images, Train_File_Labels)                # Creates an array of rotated images from original data
    #Graph(Rotate_Images)
    Train_Images = np.asarray(Rotate_Images)/255.0                                                                                  # Makes images into a numpy array and normalises the colours
    Train_Labels = np.asarray(Rotate_Labels)
    Class_weights = Summer(Train_Labels)                                                                                            # Classweights which can be used in model.fit
    
    # Slices rot images into train and validation images on a 9:1 split
    # Slice = int(round(0.9 * len(Final_Images)))
    # train_images = Final_Images[:Slice]
    # train_labels = Final_Labels[:Slice]

    # validation_images = Final_Images[Slice:]
    # validation_labels = Final_Labels[Slice:]

    print("Opening validation data set.")                                                                                           # Opens validation data (EFIGI)
    Val_Class = pd.read_csv(Val_Label_Path, delim_whitespace=True, header=None, names=["ID", "z", "RA", "Dec", "TType"])            # Validation data has no data augmentation as it is not used to train the model
    Labels = ModifiedTTypeToBinary(np.squeeze(np.array(Val_Class.loc[:,['TType']])) + 7, False)
    Val_ID = np.squeeze(np.array(Val_Class.loc[:,['ID']]))
    Val_Redshift = np.squeeze(np.array(Val_Class.loc[:,['z']]))

    Val_Images =[]
    Val_Labels =[]
    for i in range(len(Labels)):
        if(Val_Redshift[i]>0.08 or Labels[i] == 5):                                                                                 # If the redshift of a galaxy is greater than 0.08 or the galaxy is an irregular galaxy, it skips this galaxy
            continue
        else:
            Val_Images.append(image.imread(os.path.join(Val_Image_Path, str(int(Val_ID[i])) + "." + FileType)))                     # Validates images against the file and opens them to an array.
            Val_Labels.append(Labels[i])                                                                                            # If image is validate, the label is added to the label array

    Validation_Labels = np.asarray(Val_Labels)                                                                                      # Convert labels to binary form
    Validation_Images = np.asarray(Val_Images)/255.0                                                                                # Normalise colours
    print("Loaded " + str(len(Validation_Images)) + " images and " + str(len(Validation_Labels)) + " labels.")

    print("Train and validation data sets ready.")
    return Train_Images, Train_Labels, Validation_Images, Validation_Labels, Class_weights

# Redcues the size in the array of certain galaxy types (S in this case) to remove skewness in the training data
def ReduceSkew(Labels, Images, Class):
    New_Labels = []
    New_Images = []
    Summer(Labels)
    print("Image length is " + str(len(Images)) + ".Label length is " + str(len(Labels)) + ".")                     # Original image and lablel length
    for i in range(len(Labels)):
        if(Labels[i]==Class and randint(0,1)==1):                                                                   # If label is eqaul to 4 (spiral), a random number generator choses whether this label is removed from the data base
            continue
        else:
            New_Images.append(Images[i])                                                                            # If the random number generator choses 0 or the label is not eqaul to 4, the image and label are added to a new array
            New_Labels.append(Labels[i])
    
    print("New image length is " + str(len(New_Images)) + ".New label length is " + str(len(New_Labels)) + ".")     # Prints new image and label length
    Summer(New_Labels)
    return New_Labels, New_Images

# Calcultes the number of different galaxy types in the label array
def Summer(Labels):
    Counter1 = 0
    Counter2 = 0
    Counter3 = 0
    Counter4 = 0
    for i in range(len(Labels)):
        if(Labels[i] == 1):             # If galaxy is S0
            Counter1 +=1
            continue
        if(Labels[i] == 2):             # If galaxy is S0a
            Counter2 +=1
            continue
        if(Labels[i] == 3):             # If galaxy is E
            Counter3 +=1
            continue
        if(Labels[i] == 4):             # If galaxy is S
            Counter4 +=1
            continue
    Total_Sum = Counter1 + Counter2 + Counter3 + Counter4
    print("Number of S0:" + str(Counter1) + ". Number of S0a:" + str(Counter2) + ". Number of E:" + str(Counter3) + ". Number of S:" + str(Counter4)) # Prints number of galaxy types to screen
    return {1: Counter1/Total_Sum, 2: Counter2/Total_Sum, 3: Counter3/Total_Sum, 4: Counter4/Total_Sum}
