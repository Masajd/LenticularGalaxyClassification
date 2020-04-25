import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import pandas as pd

# Class Files
from LabelConverter import Binarise
from utils.metrics import*

# Variables
Method =1
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

Prob_Low = dict()
Prob_Med = dict()
Prob_High = dict()

Mean_Low = dict()
Mean_Med = dict()
Mean_High = dict()

Classes = ["SD TType 1", "SD TType 2", "SD TType 3", "SD TType 4"]
Surveys = ["GZ2", "NA10", "EFIGI"]

Data = pd.read_csv(Data_Path, delim_whitespace=True, header=None,
                names=["Asset_ID", "Class", "Predicted_Class", "Bias", "S0_Prob", "S0a_Prob", "E_Prob", "S_Prob"]) 

# Creates new classification array from 'Class'
Labels = Binarise(np.squeeze(np.array(Data.loc[:,["Class"]])))
Pred_Labels = Binarise(np.squeeze(np.array(Data.loc[:,["Predicted_Class"]])))
Probabilities = np.squeeze(np.array(Data.loc[:,["S0_Prob", "S0a_Prob", "E_Prob", "S_Prob"]]))

for i in range(len(Classes)):
    if(np.count_nonzero(Labels[:,i])):
        #Prob[i], Mean[i] = calibration_curve(Labels[:,i], Probabilities[:,i], strategy= "uniform", normalize= True )#, n_bins=int(round(len(Labels)/1000)))
        Prob_Low[i], Prob_Med[i], Prob_High[i] = get_bayes_interval(Labels[:,i], Probabilities[:,i]) 
        Mean_Low[i], Mean_Med[i], Mean_High[i] = get_interval(Probabilities[:,i])
    else:
        continue

# Add plots to figure
for i in range(len(Classes)):
    if(i in Prob_Med):
        plt.errorbar(Mean_Med[i], Prob_Med[i],
        xerr=[Mean_Med[i]- Mean_Low[i], Mean_High[i]- Mean_Med[i]],
        yerr=[Prob_Med[i]- Prob_Low[i], Prob_High[i]- Prob_Med[i]],
        label= Classes[i])
        print(calibration_error(Labels[:,i], Probabilities[:,i], int(round(len(Labels)/10))))
    else:
        continue

#Prints figure to screen
plt.plot([0, 1], [0, 1], 'k--')
plt.legend(loc=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("Mean Predicted Value", fontsize= 14)
plt.ylabel("Fraction of Positives", fontsize= 14)
plt.title("Calibration Curve from " + Surveys[Method-1], fontsize= 14)
plt.show()
#plt.savefig(r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Final Results\Curves\\"+ Surveys[Method-1] +)
