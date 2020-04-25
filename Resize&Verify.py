# Image and File Manipulation
import pandas as pd
from matplotlib import image
from skimage.transform import resize
import os
import os.path
from PIL import Image
from os import system,name

# Mathematical Functions
import numpy as np
from matplotlib import pyplot as plt

# Class Functions and Variables
from LabelConverter import TTypeToBinary, ClassificationToTType, ClassificationToBinary
WriteToDoc = False

#Opens mapping and labels file 
Map = pd.read_csv(r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\GZ2_DataV1.txt", 
                    delim_whitespace=True, 
                    header=None, 
                    names=["asset_id", "RA", "DEC", "gz2_class"])
ID = np.squeeze(np.array(Map.loc[:,['asset_id']]))
Label = ClassificationToBinary(np.squeeze(np.array(Map.loc[:,['gz2_class']])))

# Creates new text document for mapping
Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\GZ2_Map_Bin.txt"
if(WriteToDoc):
    if os.path.exists(Path):
        os.remove(Path)
    document = open(Path, "a+")
    document.write("Asset ID" + "\t" + "Binary Classification")

# Old and new directory with file type for new files. Counter is used to determine number of special galaxy types
counter = 0
percent = 0
size = 96
file_type = "jpg"
dir_name = "F:\samue\Pictures\GZ2_424"
new_dir_name = "F:\samue\Pictures\GZ2_" + str(size)

for i in range(len(ID)):

    new_percent = round((i/len(ID))*100, 2)                         # Prints a progress bar to screen
    if(new_percent>percent):
        system('cls')
        print( str(new_percent) + "%")
        percent = new_percent

    Old_dir = os.path.join(dir_name, str(ID[i]) + "." + file_type)  # Creates old file path using old directory and file type

    if(os.path.isfile(Old_dir)):                                    # If file exists, continue
        
        if(Label[i]==5):                                            # If galaxy file exists with unsatisfactory classification, skip file
            counter += 1                                            # Add one to counter for final analysis
            continue
        else:
            Open_Image = image.imread(Old_dir)                                                          # Opens image from old directory
            Cropped_Image = Open_Image[60:364, 60:364, :]                                               # Crops the image by 60 on each side
            Resized_Image = resize(Cropped_Image, (size,size), anti_aliasing=True)                      # Resize image to 64x64
            plt.imsave(os.path.join(new_dir_name, str(ID[i]) + "." + file_type), Resized_Image)         # Creates new file in new directory path
            if(WriteToDoc):
                document.write("\n" + str(ID[i]) + "\t" + str(Label[i]))                                # Writes new mapping file with ID and binary classification
    else:
        continue                                                                                        # If file doesn't exist then skip ID

print("Number of stars, edge on or irregular galaxies is " + str(counter))                              # Prints the number of special galaxies removed from the file
if(WriteToDoc):
    document.close()

