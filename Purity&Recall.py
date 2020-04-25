# Scripts
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from os import system,name
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, matthews_corrcoef, balanced_accuracy_score, log_loss, recall_score
from sklearn.preprocessing import Binarizer

from LabelConverter import Binarise

# Variables
Surveys =["GZ2", "NA10", "EFIGI"]
Classes = ["SD TType 1", "SD TType 2", "SD TType 3", "SD TType 4"]
Method = 3
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
        Data_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\N&A Results\NA_Class_Results" + Model +"V" + str(Version) + "DR" + str(DR) + str(size) + "_DR" + str(TestDR) +".txt"
    if(Method ==3):
        Data_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\EFIGI Results\EFIGI_Class_Results" + Model + "V" + str(Version)+ "DR" + str(DR) + str(size) + "_DR" + str(TestDR) +".txt"
else:
    if(Method==1):
        Data_Path = r"C:\Users\Sam Dicker\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Galaxy Zoo 2 Results\GZ_Class_Results" + Model +"V" + str(Version) + "DR" + str(DR) + str(size) + "_DR" + str(TestDR) +".txt"
    if(Method==2):
        Data_Path = r"C:\Users\Sam Dicker\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\N&A Results\NA_Class_Results" + Model +"V" + str(Version) + "DR" + str(DR) + str(size) + "_DR" + str(TestDR) +".txt"
    if(Method==3):
        Data_Path = r"C:\Users\Sam Dicker\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\EFIGI Results\EFIGI_Class_Results" + Model +"V" + str(Version) + "DR" + str(DR) + str(size) + "_DR" + str(TestDR) +".txt"

Data = pd.read_csv(Data_Path, delim_whitespace=True, header=None,
                names=["Asset_ID", "Class", "Predicted_Class", "Bias", "S0_Prob", "S0a_Prob", "E_Prob", "S_Prob"]) 

Probabilities = np.squeeze(np.array(Data.loc[:,["Class","Predicted_Class", "S0_Prob", "S0a_Prob", "E_Prob", "S_Prob"]]))
Prob=[]
for i in range(len(Probabilities)):
    if(Probabilities[i][1]==1):
        if(Probabilities[i][2]>0.6):
            Prob.append(Probabilities[i])
            continue
        else:
            continue

    if(Probabilities[i][1]==3):
        if(Probabilities[i][4]>0.6):
            Prob.append(Probabilities[i])
            continue
        else:
            continue

    if(Probabilities[i][1]==4):
        if(Probabilities[i][5]>0.6):
            Prob.append(Probabilities[i])
            continue
        else:
            continue

    if(Probabilities[i][1]==2):
        continue
print(np.squeeze(Prob))
Labels = np.squeeze(Binarise(np.squeeze(Prob)[:,0]))
print(Labels)
Pred_Labels = np.squeeze(Binarise(np.squeeze(Prob)[:,1]))
print(Pred_Labels)
print(matthews_corrcoef(Labels, Pred_Labels))
# Probabilities = np.squeeze(np.array(Data.loc[:,["S0_Prob", "S0a_Prob", "E_Prob", "S_Prob"]]))
# Labels = Binarise(np.squeeze(np.array(Data.loc[:,["Class"]])))
# Pred_Labels = Binarise(np.squeeze(np.array(Data.loc[:,["Predicted_Class"]])))
# Purity = dict()
# Recall = dict()
# Avg_Pre = dict()
# Avg_Recall = dict()
# MattCo = dict()
# Accuracy = dict()
# Loss = dict()

# for i in range(len(Classes)):
#     if(np.count_nonzero(Labels[:,i])):
#         Purity[i], Recall[i], _ = precision_recall_curve(Labels[:,i], Probabilities[:,i])
#         Avg_Pre[i] = average_precision_score(Labels[:,i], Probabilities[:,i])
#         Avg_Recall[i] = recall_score(Labels[:,i], Pred_Labels[:,i])
#         MattCo[i] = matthews_corrcoef(Labels[:,i], Pred_Labels[:,i])
#         Accuracy[i] = balanced_accuracy_score(Labels[:,i], Pred_Labels[:,i])
#         Loss[i] = log_loss(Labels[:,i], Probabilities[:,i])
#     else:
#         continue

# # Create graph figure
# plt.figure()#figsize=(10,8))

# # Add plots to figure
# for i in range(len(Classes)):
#     if(i in Purity):
#         plt.plot(Purity[i], Recall[i], label= Classes[i]) #+ "(Avg Precision = %0.2f)"%Avg_Pre[i])
#         print(str(Classes[i]) + " : " + str(Accuracy[i]) +", "+ str(Loss[i]) + ", " + str(MattCo[i]) + ", " + str(Avg_Pre[i]) + ", " + str(Avg_Recall[i]))
#     else:
#         continue

# #Prints figure to screen
# if(Method ==3):
#     plt.legend(loc=(0.20,0.65))
# else:
#     plt.legend(loc=3)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel("Precision", fontsize= 14)
# plt.ylabel("Recall", fontsize= 14)
# plt.title("Precision/Recall Curve from " + Surveys[Method-1], fontsize= 14)
# plt.show()