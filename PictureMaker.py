# Maths Functions
import numpy as np
import math
from random import randint

# File manipulation
import os, os.path
import gc
import pandas as pd
import glob
from matplotlib import image, pyplot as plt

# Class Files
from ImageManipulation import Graph

size=96
DR = 16
SDT =4
if(SDT==4):
    Probability = 5
if(SDT==3):
    Probability = 4
if(SDT==1):
    Probability = 3

Image_Path = r"F:\samue\Pictures\GZ2_" + str(size) + "_DR" + str(DR)
Map_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Final Results\Class and Probs\GZ_FullData_V3.1.1.txt"  

Map = pd.read_csv(Map_Path, delim_whitespace=True)#, header=None, names=["id", "z", "RA", "DEG", "class"])    # Opens a mapping file 
Data = np.squeeze(np.array(Map.loc[:,["Asset_ID","LGM_TOT_P50", "Predicted_Class", "S0_Prob", "E_Prob", "S_Prob", "SFR_TOT_P50"]]))

Galaxies = Data[Data[...,2]==SDT]
Prob_Cut = Galaxies[np.logical_and(Galaxies[...,Probability]<0.55, Galaxies[...,Probability]>0.5)]
Mass_Range = Prob_Cut[np.logical_and(Prob_Cut[...,1]>10.5, Prob_Cut[...,1]<10.8)]
SFR_Cut = Mass_Range[Mass_Range[...,6]>-0.5][:,[0]]
#ID_Low_S= np.delete(Mass_S, np.argwhere(Mass_S[:,[1]]>8.5), axis=0)[:,[0]]
np.random.shuffle(SFR_Cut)

print(len(SFR_Cut))
if(len(SFR_Cut)<36):
    Limit = len(SFR_Cut)
else:
    Limit = 36
Images = []
for i in range(Limit):
    Images.append(image.imread(os.path.join(Image_Path, str(int(SFR_Cut[i])) + ".jpg")))
Graph(Images, Limit)