from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Class Files
from LabelConverter import Binarise

# Variables
Method =2
size = 96
DR = 16
TestDR = 16
Version = 3.1
Complex_Model = True
Main_PC= True
if(Complex_Model):
    Model = "_Feature"
else:
    Model = "_"
if(Main_PC):
    if(Method ==1):
        Data_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Galaxy Zoo 2 Results\GZ_Class_Results" + Model +"V" + str(Version) + "DR" + str(DR) + str(size) + "_DR" + str(TestDR) +".txt"
    if(Method ==2):
        Data_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\N&A Results\NA_Class_Results" + Model +"V" + str(Version) + "DR" + str(DR) + str(size) + "_DR" + str(TestDR) +"_Central.txt"
    if(Method ==3):
        Data_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\EFIGI Results\EFIGI_Class_Results" + Model + "V" + str(Version)+ "DR" + str(DR) + str(size) + "_DR" + str(TestDR) +".txt"
else:
    if(Method==1):
        Data_Path = r"C:\Users\Sam Dicker\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Galaxy Zoo 2 Results\GZ_Class_Results" + Model +"V" + str(Version) + "DR" + str(DR) + str(size) + "_DR" + str(TestDR) +".txt"
    if(Method==2):
        Data_Path = r"C:\Users\Sam Dicker\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\N&A Results\NA_Class_Results" + Model +"V" + str(Version) + "DR" + str(DR) + str(size) + "_DR" + str(TestDR) +".txt"
    if(Method==3):
        Data_Path = r"C:\Users\Sam Dicker\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\EFIGI Results\EFIGI_Class_Results" + Model +"V" + str(Version) + "DR" + str(DR) + str(size) + "_DR" + str(TestDR) +".txt"

FPR = dict()
TPR = dict()
ROC_AUC = dict()
Classes = ["SD TType 1", "SD TType 2", "SD TType 3", "SD TType 4"]
Surveys = ["GZ2", "NA10", "EFIGI"]

Data = pd.read_csv(Data_Path, delim_whitespace=True, header=None,
                names=["Asset_ID", "Class", "Predicted_Class", "Bias", "S0_Prob", "S0a_Prob", "E_Prob", "S_Prob"]) 

# Creates new classification array from 'Class'
Probabilities = np.squeeze(np.array(Data.loc[:,["S0_Prob", "S0a_Prob", "E_Prob", "S_Prob"]]))
Labels = Binarise(np.squeeze(np.array(Data.loc[:,["Class"]])))

for i in range(len(Classes)):
    if(np.count_nonzero(Labels[:,i])):
        FPR[i], TPR[i], _ = roc_curve(Labels[:,i], Probabilities[:,i])
        ROC_AUC[i] = auc(FPR[i], TPR[i])
    else:
        continue

plt.figure()#figsize=(10,8))
for i in range(len(Classes)):
    if(i in FPR):
        plt.plot(FPR[i], TPR[i], label= Classes[i] + ' (AUC = %0.2f)'% ROC_AUC[i])
    else:
        continue
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize= 14)
plt.ylabel('True Positive Rate', fontsize= 14)
plt.title('ROC from testing ' + Surveys[Method-1] + ' Galaxies', fontsize= 14)
plt.legend(loc="lower right")
plt.show()
