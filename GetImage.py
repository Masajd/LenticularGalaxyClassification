import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image
from scipy import ndimage
import math
import urllib
import os
from os import system,name
import socket
import random
import gc
from LabelConverter import ClassificationToBinary
# Variables
Method =1                       # Which ra and dec study to use (1= GZ2, 2= NA10 and 3= EFIGI)
size = 96                       # The size of the image that be downloaded (size x size pixels)
Centered = True                 # Controls whether galaxies are in the center of the images or distributed around the center
DR = 16                         # Which data release of the SDSS (Sloan Digital Sky Survey) to use
Main_PC = False                 # Changes inital path depending on which PC this script is run on

# Script Variables
gc.collect()                    # Removes all unused previous files from ram
percent = 0
Counter = 0                     # To count how many unclassifiable galaxies are removed
socket.setdefaulttimeout(60)    # Calls urllib to move on after 60 seconds of no response
if(Main_PC):
    Map_Path = r"F:"
    Image_Path = r"F:\samue"
else:
    Map_Path = Image_Path = r"C:\Users\Sam Dicker"

# Imports data file of Galaxy IDs, redshift, TType, RA and DEC
if(Method==1):
    Map = pd.read_csv(Map_Path + r'\\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\GZ2_DataV1.txt', delim_whitespace=True, header=None, names=["ID", "z", "RA", "DEC", "GZ2Class"])
    Class = ClassificationToBinary(np.squeeze(np.array(Map.loc[:,["GZ2Class"]])))
    Redshift = np.squeeze(np.array(Map.loc[:,["z"]]))
    Image = Image_Path + r"\\Pictures\GZ2_" + str(size) + "_DR" + str(DR)
if(Method==2):
    Map = pd.read_csv(Map_Path + r'\\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\AbrahamDatav3.txt', delim_whitespace=True, header=None, names=["ID", "z", "RA", "DEC", "TType"])
    if(Centered):
        Image = Image_Path + r"\\Pictures\Galaxies_" + str(size) + "_Central_DR" + str(DR) 
    else:
        Image = Image_Path + r"\\Pictures\Galaxies_" + str(size) + "_DR" + str(DR)
if(Method==3):
    Map = pd.read_csv(Map_Path + r'\\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\EFIGI_DataV1.txt', delim_whitespace=True, header=None, names=["ID", "z", "RA", "DEC", "TType"])
    Image = Image_Path + r"\\Pictures\EFIGI_" + str(size) + "_DR" + str(DR)
urls = []

# Creates urls for download. Uses redshift value to scale image, and to adjust RA and DEC so the galaxy isn't in the center of the image. The downloaded image will be 64 x 64 image.  
for i in range(len(Map)):
    #scale = -8.0350877*Map.loc[i][1] + 1.399825                # Function that converts redshift value into a scale
    #RA_DEC = -0.013671875*Map.loc[i][1] + 0.0044296875         # Function that converts redshift into a range that RA and DEC can be changed by
    # scale = -8.0350877*Map.loc[i][1] + 1.009825               # 128 images need different ranges
    # RA_DEC = -0.013671875*Map.loc[i][1] + 0.0031296875
    scale = -8.0350877*Map.loc[i][1] + 0.8590375                # Galaxy Zoo 2 images are small
    RA_DEC = -0.013671875*Map.loc[i][1] + 0.0018296875

    if(DR==16): # DR14 or DR16 url (Change this -----¬)
        if(Centered):                              # v
            urls.append("http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Chart.Image&ra=" + str(Map.loc[i][2]) + "&dec=" +   
                        str(Map.loc[i][3]) + "&scale=" + str(scale) + "&width=" + str(size) + "&height=" + str(size) + "&opt=&query=")
        else:
            urls.append("http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Chart.Image&ra=" + str(Map.loc[i][2] + random.uniform(-RA_DEC, RA_DEC)) + "&dec=" +  # Random uniform controls the distribution of the galaxies around the center of the image
                        str(Map.loc[i][3] + random.uniform(-RA_DEC, RA_DEC)) + "&scale=" + str(scale) + "&width=" + str(size) + "&height=" + str(size) + "&opt=&query=")
    if(DR==7):  # DR7 
        if(Centered):
            urls.append("http://skyservice.pha.jhu.edu/DR7/ImgCutout/getjpeg.aspx?ra=" + str(Map.loc[i][2]) + "&dec=" + 
                        str(Map.loc[i][3]) + "&scale=" + str(scale) + "&width=" + str(size) + "&height=" + str(size) + "&opt=&query")
        else:
            urls.append("http://skyservice.pha.jhu.edu/DR7/ImgCutout/getjpeg.aspx?ra=" + str(Map.loc[i][2] + random.uniform(-RA_DEC, RA_DEC)) + "&dec=" + 
                        str(Map.loc[i][3] + random.uniform(-RA_DEC, RA_DEC)) + "&scale=" + str(scale) + "&width=" + str(size) + "&height=" + str(size) + "&opt=&query")

# Downloads the images from the urls, and adds their ID as their jpeg name. If the image already exists or the galaxy has a redshift greater than 0.08, 
# then the loop skips that file.
for i in range(len(urls)):
    
    if(Method==1 and Class[i] ==5):                                                         # If the galaxy is unclassifiable, a star etc it is removed
        Counter +=1
        continue
    if(Method==1 and Redshift[i]>0.08):                                                     # If the galaxy has a redshift greater than 0.08, 
        continue                                                                            # it is too small and removed.

    new_percent = round((i/len(urls))*100, 2)                                               # Prints a progress bar to screen
    if(new_percent>percent):
        system('cls')
        print( str(new_percent) + "%")
        percent = new_percent

    Batch_Path = os.path.join(Image, str(int(Map.loc[i][0])) + ".jpg")          
    if(os.path.exists(Batch_Path)):                                                         # If the file already exists, skip it
        continue
    else:
        try:
            urllib.request.urlretrieve(urls[i], Batch_Path)                                 # Downloads image, then names and gives it a file type
        except:
            urllib.request.urlretrieve("http://skyservice.pha.jhu.edu/DR7/ImgCutout/getjpeg.aspx?ra=" + str(Map.loc[i][2]) + "&dec=" +                  # Some galaxies do not exist in the DR14 or 16 data base, but do in the DR7.
                                        str(Map.loc[i][3]) + "&scale=" + str(scale) + "&width=" + str(size) + "&height=" + str(size) + "&opt=&query",   # This try: except: catches HTTP error 404 for these galaxies.
                                        Batch_Path)
        urllib.request.urlcleanup()                                                         # Clears all temporary internet files to stop "Too many calls" errors (Putting urllib requests in loops is bad practice but hey it works)                                                                                                    
print("Done!")
print("Removed " + str(Counter) +" Galaxies.")