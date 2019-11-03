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

def euclidean_distance_loss(y_true, y_pred):
    """
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

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

def get_siamese_vgg16(input_shape):

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    vgg16_model = keras.applications.vgg16.VGG16()
    model = Sequential()
    #copy the vgg (not last layer) and create own 
    for layer in vgg16_model.layers[:-1]:
        model.add(layer)
        #model.summary()
   
    #freeze wheights
    for layer in model.layers:
        layer.trainable = False 
    

    #fine-tuning
    #add layer to create an array of 4096 dimension
    model.add(Dense(units=4096, activation='sigmoid'))

    #model.summary()

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)


    '''
    #start predict 
    # load an image from file
    image1 = load_img(input1_path, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image1 = img_to_array(image1)
    # reshape data for the model
    image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
    # prepare the image for the VGG model
    image1 = preprocess_input(image1)
    # predict the probability across all output classes
    print(1)
    yhat = model.predict(image1)
    print(yhat)
    '''
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])    

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = model(inputs=[left_input,right_input],outputs=prediction)    
    # return the model
    return siamese_net    

    model.summary()
    






def main(): 
    model = get_siamese_vgg16 ((224,224,3))
    #model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],optimizer='sgd')

if __name__ == "__main__":
    main()

