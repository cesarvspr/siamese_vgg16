import numpy as np
import keras
from keras import backend as K
from keras.models import Model, Sequential
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



def euclideanDistance(base, teste):
    var = base-teste
    var = np.sum(np.multiply(var, var))
    var = np.sqrt(var)
    return var

def triplet_loss(embeddings):
    """
    calculates triplet loss over inputs.
    """
    
    processed_a, processed_p, processed_n = embeddings[0], embeddings[1], embeddings[2]
    
    positive_dist= euclidean_distance_loss(processed_a, processed_p)
    negative_dist = euclidean_distance_loss(processed_a, processed_n)
       
    margin = 0.0
    loss = K.maximum(margin, positive_dist - negative_dist)
    
    return K.mean(loss)

def read(nome):
    # load an image from file
    image = load_img(nome, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image 

def main():

    model = Sequential()

    

    vgg_model = VGGFace(include_top=True, input_shape=(224,224,3))
    vgg_model.summary()
    print('----------------------------------------')
    
    print(vgg_model.layers[24].output)
    print(vgg_model.layers[-2].output)

    model = Model(inputs=vgg_model.layers[0].input, outputs=vgg_model.layers[24].output)

    return(model.predict(read('./image.jpeg')))


'''
    pred = last.predict(read('./image.jpeg'))
    preds2 = last.predict(read('./image2.jpeg'))

    print(pred)
    print(preds2)

    last_layer = vgg16_model.get_layer('avg_pool').output
    out = Flatten(name='flatten')(last_layer)
    preds = model.predict(x)
'''

if __name__ == "__main__":
    main()

