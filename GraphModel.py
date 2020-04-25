from keras.utils import plot_model
from keras.models import load_model

Simple_Model = False                                                                            # Whether to use a simple or complex model
size = 96                                                                                       # Size of images to open
DR = 16                                                                                         # The data release of the training images
Version = 3.2                                                                                   # Model version
if(Simple_Model):
    Model_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Models\GalaxyClass_V" + str(Version) +"_DR" + str(DR) + str(size) + ".h5"          # Model path
else:
    Model_Path = r"F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Data\Models\GalaxyClass_FeatureV" + str(Version) +"_DR" + str(DR) + str(size) + ".h5"

plot_model(load_model(Model_Path), to_file=r'F:\OneDrive\Documents\Physics 4th Year\MPHYS Porject\Report\Figures\model.png')