from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import keras.backend as K
import cv2
import xlrd
import os
import numpy as np
from VFOCT_DataLoad import *
from VFOCT_Model import *
from matplotlib import pyplot as plt


# Setup ===============================================================================
image_file = "testset/922052712_20190131_OD.jpg"
output_folder = "output/CAM"
weight_file = "weights/inceptionV3FinalTrained.hdf5"
# =====================================================================================

img_pos = [                        [3, 0], [4, 0], [5, 0], [6, 0],
                           [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1],
                   [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2],
           [0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3],         [8, 3],
           [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4],         [8, 4],
                   [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [8, 5],
                           [2, 6], [3, 6], [4, 6], [5, 6], [6, 6], [7, 6],
                                   [3, 7], [4, 7], [5, 7], [6, 7]]


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def visualize_class_activation_map(model, img_path, output_path):
    original_img = cv2.imread(img_path, 1)
    width, height, _ = original_img.shape

    # Reshape to the network input shape (3, w, h).
    img = np.array([np.transpose(np.float32(original_img), (0, 1, 2))])

    # Get the input weights to the bottleneck
    class_weights4 = model.layers[-4].get_weights()[0]
    class_weights3 = model.layers[-3].get_weights()[0]
    class_weights2 = model.layers[-2].get_weights()[0]
    class_weights1 = model.layers[-1].get_weights()[0]
    class_weights = np.matmul(np.matmul(np.matmul(class_weights4, class_weights3), class_weights2), class_weights1)

    final_conv_layer = get_output_layer(model, "mixed10")
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_img = [gray_img, gray_img, gray_img]
    gray_img = np.transpose(gray_img, (1, 2, 0))

    # Create the class activation map.
    fig, axs = plt.subplots(8, 9, figsize=(322, 161))
    fig.tight_layout()
    for t in range(0, 52):
        out_file_name = "CAM_%2d_%s" % (t+1, os.path.split(image_file)[1])

        cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
        target_class = t
        for i, w in enumerate(class_weights[:, target_class]):
            cam += w * conv_outputs[:, :, i]

        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        img = heatmap * 0.5 + gray_img
        cv2.imwrite(os.path.join(output_path, out_file_name), img)   # save each CAM images
        axs[img_pos[t][1], img_pos[t][0]].imshow(load_img(os.path.join(output_path, out_file_name)))

    for r in range(0, 8):
        for c in range(0, 9):
            axs[r, c].axis('off')

    # fig.savefig("{0}/{1}_{2}.jpg".format(output_folder, "CAM_Comb", os.path.splitext(os.path.split(image_file)[1])[0]))
    plt.show()


# model build ================================
model = GetModel()
model.load_weights(weight_file)

# draw CAM
visualize_class_activation_map(model, image_file, output_folder)

