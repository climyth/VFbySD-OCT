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
import xlrd
import numpy as np


# Data loading function ===============================
def LoadData(image_folder, vf_file, image_read_from_previous_numpy=False, thv_col_start=6):
    worksheet = xlrd.open_workbook(vf_file).sheet_by_name("Train")
    nrows = worksheet.nrows - 1
    y_data = np.empty([nrows - 1, 52])
    x_data = np.empty([0, 161, 322, 3])  # OCT image shape = 322 x 161

    for r in range(1, nrows):
        if image_read_from_previous_numpy == False:
            img_filename = worksheet.cell_value(r, 0)
            img = load_img(image_folder + "/" + img_filename)
            x_img = img_to_array(img)
            x_img = x_img.reshape((1,) + x_img.shape)
            x_data = np.concatenate((x_data, x_img), axis=0)
            print("[%d] concatenated: %s" % (r, img_filename))
        c1 = 0
        for c in range(0, 54):
            if c != 25 and c != 34:  # physiological scotoma excluded
                y_data[r - 1, c1] = worksheet.cell_value(r, c + thv_col_start)
                c1 = c1+1

    if image_read_from_previous_numpy:
        x_data = np.load(image_folder+"/img_data.npy")
    else:
        np.save(image_folder + "/img_data", x_data)
        print("Image array saved")

    print("Loading completed. X data shape is %s"%((x_data.shape,)))

    return x_data, y_data

