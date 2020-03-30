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
import numpy as np


def GetModel():
    # model build ================================
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(161, 322, 3))
    # base_model = VGG19(weights='imagenet', include_top=False, input_shape=(161, 322, 3))
    # base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(161, 322, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    # and a logistic layer -- let's say we have 54 classes
    predictions = Dense(52, activation='relu')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

