from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, GlobalAveragePooling2D
import xlrd
import numpy as np
from VFOCT_DataLoad import *
from VFOCT_Model import *

# Setup =============================================================
image_folder = ""   # root image folder for train set
vf_file = "VFTrain.xlsm"   # visual field data excel file
weight_save_folder = "Weights"
graph_save_folder = ""   # model graph output folder
pretrained_weights = ""   # if no pretrained weight, just leave ""
tensorboard_log_folder = "logs"
# ===================================================================


# Data loading ===============================
print("Data loading...")
x_train, y_train = LoadData(image_folder, vf_file, True)

# model build ================================
model = GetModel()
if pretrained_weights != "":
    model.load_weights(pretrained_weights)
# plot_model(model, to_file=graph_save_folder+"/graph.png", show_shapes=True, show_layer_names=True)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='mean_squared_error')

# checkpoint
filepath = weight_save_folder + "/inceptionV3_weights-improvement-{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir=tensorboard_log_folder, histogram_freq=1, write_graph=True, write_images=True)
callbacks_list = [checkpoint, tensorboard]

# Train ===========================================
model.fit(x_train, y_train, batch_size=64, validation_split=0.1, epochs=10000, callbacks=callbacks_list)

