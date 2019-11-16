import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.utils.vis_utils import plot_model
import keras_vggface
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras import layers
from keras.layers import Flatten


def main():

    nome = '/home/notme2/Documents/siamese_vgg16/image.jpeg'
    print(55)
    # load an image from file
    image = load_img(nome, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    print(image)
    return image

    


if __name__ == "__main__":
    main()
