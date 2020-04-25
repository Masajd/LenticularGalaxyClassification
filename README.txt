You'll need to edit the paths to make these work. File extraction is set up for text files not CSV files.

NeuralNetwork - For training and validating the network, saves network to drive. 

ImageTest - For testing the network, saves probabilities to a file on drive. 

ImageManipulation - Contains Training Image Loader as well as cropper, rotator, randomiser and various other techniques to change images

LabelConverter - Contains definitions to change labels from one Classification Scheme to your own scheme. Has been set up to change schemes to SD TType scheme at the moment

ROC_Curve, Puirty&Recall, and Calibration Curve - Produce ROC, Purity & Recall and Calibration cuvres from the probability file prodcued by ImageTest

GetImage - Downloads images from SDSS website

Resize&Verify - Changes the size of images, and verifies it exists in your label file. Dont use this, it produces blurry images and the image loaders verify your images anyway.