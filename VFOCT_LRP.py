import innvestigate
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, GlobalAveragePooling2D
import xlrd
import numpy as np
from VFOCT_DataLoad import *
from VFOCT_Model import *
from matplotlib import pyplot as plt
import os


# Setup =============================================================================
image_file = "testset/922052712_20190131_OD.jpg"
output_folder = "output/LRP"
weight_file = "weights/inceptionV3FinalTrained.hdf5"
# ====================================================================================

img_pos = [                        [3, 0], [4, 0], [5, 0], [6, 0],
                           [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1],
                   [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2],
           [0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3],         [8, 3],
           [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4],         [8, 4],
                   [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [8, 5],
                           [2, 6], [3, 6], [4, 6], [5, 6], [6, 6], [7, 6],
                                   [3, 7], [4, 7], [5, 7], [6, 7]]

# make deep learning model
model = GetModel()
model.load_weights(weight_file)

# load OCT image
img = load_img(image_file)
x_img = img_to_array(img)
x_img = x_img.reshape((1,) + x_img.shape)

# get visual field prediction
pred = model.predict(x_img)
print("Predicted 52 THV values")
print(pred)

# LRP (Layer-wise relevance propagation)
analyzer = innvestigate.create_analyzer("lrp.sequential_preset_a", model, neuron_selection_mode="index")
fig, axs = plt.subplots(8, 9, figsize=(322, 161))
fig.tight_layout()

for i in range(0, 52):
    a = analyzer.analyze(x_img, i)
    # Aggregate along color channels and normalize to [-1, 1]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))
    # Plot
    axs[img_pos[i][1], img_pos[i][0]].imshow(a[0], cmap="seismic", clim=(-1, 1))

for r in range(0, 8):
    for c in range(0, 9):
        axs[r, c].axis('off')


# fig.savefig("{0}/{1}_{2}.jpg".format(output_folder, "LRP", os.path.splitext(os.path.split(image_file)[1])[0]))
plt.show()

