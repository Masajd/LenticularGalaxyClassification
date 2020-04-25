# Mathmatical Functions
import numpy as np
import math
import os

# Converts a TType to a rudimentry binary classification system consisting of 1,2,3,4 which are equal to S0, S0a, E and S respectively          # Ranges are TType ranges of each galaxy type, change the number in Binary.append to accommodate different TType scheme. i.e 1 to 0 for lenticulars on line 13
                                                                                                                                                # You will neeed to change ModifiedTTypeToBinary, ClassificationToBinary, and Binarise. All other definitions are old versions. 
                                                                                                                                                # (You could swap my binarise out with the sklearn one)
def ModifiedTTypeToBinary(Class, Text):
    
    Binary = []
    for i in range(len(Class)):

        if(Class[i] in range(4, 6)):        # Lenticular Galaxy
            Binary.append(1)
            continue

        if(Class[i] ==6):                   # Lenticular Galaxy
            Binary.append(1)
            continue

        if(Class[i] == 7):
            Binary.append(2)                # Lenticular galaxy (S0/a)
            continue

        if(Class[i] in range(0, 3)):        # Elliptical Galaxy
            Binary.append(3)
            continue

        if(Class[i] == 3):                  # Elliptical Galaxy
            Binary.append(3)
            continue

        if(Class[i] in range(8, 15)):
            Binary.append(4)                # Spiral Galaxy
            continue

        if(Class[i] == 15):
            Binary.append(4)                # Spiral Galaxy
            continue

        if(Class[i] in range(16, 20)):
            Binary.append(5)                # Irregular or edge-on galaxy
            continue
            
        if(Class[i] in range(97, 107)):
            Binary.append(5)                # Star
            continue
        
        print("Mistake on step " + str(i) + "\t" +str(Class[i]))
    if(Text):
        np.savetxt(r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Labels.txt", np.concatenate((np.array([Class]).T, np.array([Binary]).T), axis=1), delimiter= "\t", newline="\n", fmt= "%i %i")
    return Binary

def TTypeToBinary(Class):
    
    Binary = []
    for i in range(len(Class)):

        if(Class[i] in range(-1,-3, -1)):   # Lenticular Galaxy (S0 minus to S0 plus)
            Binary.append(1)
            continue

        if(Class[i] in range(-3,-1)):       # Lenticular Galaxy (Doesnt like minuses)
            Binary.append(1)
            continue

        if(Class[i] == 0):
            Binary.append(2)                # Lenticular galaxy (S0/a)
            continue

        if(Class[i] in range(-4, -7, -1)):
            Binary.append(3)                # Elliptical Galaxy 
            continue

        if(Class[i] in range(-7,-4)):       # Elliptical Galaxy (Doesnt seem to like minuses)
            Binary.append(3)
            continue

        if(Class[i] in range(1,13)):
            Binary.append(4)                # Spiral Galaxy
            continue

        if(Class[i] in range(90,100)):
            Binary.append(5)                # Star, Irregular or edge-on galaxy
            continue

    #np.savetxt(r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Labels.txt", np.array(Binary), delimiter= "\t", newline="\n", fmt= "%i")
    return Binary

# Converts a hubble classification to TTypes
def ClassificationToTType(Class):

    TType = []
    for i in range(len(Class)):
        
        length = len(Class[i])              # Length of string needed to stop errors with different string lengths. The first 2 or 3 letters gives us the basic TType depending on whether its barred.

        if(Class[i][0] in ["S"]):           # Spiral

            if(Class[i][1] in ["a"]):       # Sa spiral galaxy given a TType of 1
                TType.append(1)
                continue

            if(Class[i][1] in ["b"]):       # Sb spiral galaxy given a TType of 3
                TType.append(3)
                continue

            if(Class[i][1] in ["c"]):       # Sc spiral galaxy given a TTYpe of 5 
                TType.append(5)
                continue

            if(Class[i][1] in ["d"]):       # Sd spiral galaxy given a TType of 7
                TType.append(7)
                continue

            if(length >2):
                
                if(Class[i][2] in ["a"]):   # SBa spiral galaxy given a TType of 1
                    TType.append(1)
                    continue
                
                if(Class[i][2] in ["b"]):   # SBb spiral galaxy given a TType of 3
                    TType.append(3)
                    continue
                
                if(Class[i][2] in ["c"]):   # SBc spiral galaxy given a TType of 5
                    TType.append(5)
                    continue

                if(Class[i][2] in ["d"]):   # SBd spiral galaxy given a TType of 7
                    TType.append(7)
                    continue

                if(Class[i][1] in ["e"] and Class[i][2] in ["r", "b", "n"]): # Round, boxy or nan edge-on spiral galaxy
                    TType.append(98)
                    continue

        if(Class[i][0] in ["E"]):           # Elliptical

            if(Class[i][1] in ["c"]):       # Cigar shaped elliptical galaxy
                TType.append(-4)
                continue

            if(Class[i][1] in ["i"]):       # Inbetween elliptical galaxy
                TType.append(-5)
                continue
            
            if(Class[i][1] in ["r"]):       # Boxy elliptical galaxy
                TType.append(-6)
                continue

        if(Class[i][0] in ["A"]):           # A star
            TType.append(99)
            continue

        if(Class[i][0] in ["Irr"]):         # Irregular
            TType.append(97)
            continue

    return TType

# Turns a hubble classification into a rudimentry binary system
def ClassificationToBinary(Class):
    
    Binary =[]
    for i in range(len(Class)):
        
        if(Class[i][0] == "S" and Class[i][1] == "e" and Class[i][2] in ["r", "n", "b"]):   # Round, boxy or nan edge-on spiral galaxy
            Binary.append(5)
            continue
        if(Class[i][0] == "S" and Class[i][1] in ["B", "a", "b", "c", "d"] ):               # Spiral
            Binary.append(4)
            continue
        if(Class[i][0] == "E" and Class[i][1] in ["c", "r", "i"]):                          # Elliptical
            Binary.append(3)
            continue
        if(Class[i][0] == "A"):                                                             # Star
            Binary.append(5)
            continue
        else:
            print("Eroooor")

    return Binary

# Combines 4 arrays into 1 to be saved to a text file.
def FourArrayToArray(ID, Class, Predicted_Class, Probability):
    
    Combined_Array = np.concatenate((np.array([ID]).T, np.array([Class]).T, np.array([Predicted_Class]).T, Probability), axis=1)

    return Combined_Array

# Turns a classification system into a binary one vs all setup.
def Binarise(Class):

    Labels = np.empty((len(Class),0))               # Creates an empty array to append new labels to
    for j in range(1):                              # The new array will have a x-axis size of 4 as there are 4 classes
        Temp_Labels=[]
        if(j == 0):
            for i in range(len(Class)):             # S0 galaxy binarisation. If class equals 1 then new label is 1, else it is now 0 
                if(Class[i]==1):
                    Temp_Labels.append(1)
                else:
                    Temp_Labels.append(0)
        
        if(j == 1):
            for i in range(len(Class)):             # S0a galaxy binarisation. If class equals 2 then new label is 1, else it is now 0
                if(Class[i]==2):
                    Temp_Labels.append(1)
                else:
                    Temp_Labels.append(0)
        
        if(j == 2):
            for i in range(len(Class)):             # E galaxy binarisation. If class equals 3 then new label is 1, else it is now 0
                if(Class[i]==3):
                    Temp_Labels.append(1)
                else:
                    Temp_Labels.append(0)
        
        if(j == 3):
            for i in range(len(Class)):             # S galaxy binarisation. If class equals 4 then new label is 1, else it is now 0
                if(Class[i]==4):
                    Temp_Labels.append(1)
                else:
                    Temp_Labels.append(0)

        Labels = np.append(Labels, np.array([Temp_Labels]).T, axis=1)

    return Labels