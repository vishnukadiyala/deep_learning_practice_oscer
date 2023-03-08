# required packages: tensorflow, keras, pandas, numpy, matplotlib, sklearn
import tensorflow as tf 
import sys
import argparse
import pickle
import pandas as pd
import numpy as np


# Model building packages
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from keras.layers import GlobalMaxPooling2D
from keras.layers import SpatialDropout2D
from keras.layers import InputLayer

#   Create parser
args = argparse.ArgumentParser()

# Model Building function with default parameters
def create_cnn_classifier_network(
    image_size,
    nchannels,
    conv_layers = None,
    dense_layers = None,
    p_dropout = None,
    p_spatial_dropout = None,
    lambda_l2 = None,
    lambda_l1 = None,
    lrate=0.001,
    n_classes = 10,
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
    padding = 'same',
    flatten = True,
    args = args):

    '''
    This part is used to update kernel with regularization if regularization exists
    so that we can add the regularization to the model 
    
    I checked if kernel = None, then no regularization exists and we can add the model without regularization
    This had no issues with the model. Hence, proceeded with the model building.
    '''
    # Check if regularization exists, then update kernel with regularization
    #if either lambda_l2 or lambda_l1 exists, then uodate kernel with regularization:
    if (lambda_l2 is not None) or (lambda_l1 is not None):
        
        #if both lambda_l2 and lambda_l1 exist, then update kernel with l1_l2 regularization:
        if (lambda_l2 is not None) and (lambda_l1 is not None):
            kernel = tf.keras.regularizers.l1_l2(lambda_l1, lambda_l2)
        else:
            #if only lambda_l1 exists, then update kernel with l1 regularization:
            if lambda_l1 is not None:
                kernel = tf.keras.regularizers.l1(lambda_l1)
            
            #if only lambda_l2 exists, then update kernel with l2 regularization:
            if lambda_l2 is not None:
                kernel = tf.keras.regularizers.l2(lambda_l2)
    #set kernel to None if no regularization exists
    else:
        kernel = None


    '''
    This is the model building part.
    
    1. Convolutional Layers
    We add the input layer with image size and number of channels. then we add spatial dropout if exists.
    Once the input is set up we will use the layers in conv_layers to add convolutional layers.
    And we Max Pool after each convolutional layer.
    and then we add Global Max Pooling Layer to find the most important features.
    
    2. Dense Layers
    We add dense layers based on the layers in dense_layers.
    add dropout if exists.
    kernel is added and has been taken care of prior to this step.
    '''
    # Create model
    model = tf.keras.Sequential()
    
    # Add Input Layer with image size and number of channels, also add Spatial Dropout if exists
    model.add(InputLayer(input_shape=(image_size[0],image_size[1], nchannels)))
    if p_spatial_dropout is not None: 
        model.add(SpatialDropout2D(p_spatial_dropout))
    
    # Add additional convolutional layers based on conv_layers input file
    for i, n in enumerate(conv_layers):
        model.add(Conv2D(n['filters'], n['kernel_size'],padding = padding, name = 'conv_{}'.format(i)))
        
        # Add Spatial Dropout if exists
        if p_spatial_dropout is not None: 
            model.add(SpatialDropout2D(p_spatial_dropout))
            
        #Add Max Pooling if exists
        # We will add Max Pooling after each convolutional layer only if pool_size > 1. Used this condition to give an input from text file
        if n['pool_size'] > 1:    
            model.add(MaxPooling2D(pool_size = n['pool_size'], strides = n['strides'], name = 'pool_{}'.format(i+1)))
    
    # Global Max Pooling Layer to find the most important features
    model.add(GlobalMaxPooling2D())
    
    # Add dense layers
    
    for i,n in enumerate(dense_layers):
        
        #add dense layer with kernel regularization
        model.add(Dense(n['units'], activation = 'relu', kernel_regularizer = kernel, name = 'dense_{}'.format(i+1)))
        
        # Add dropout if exists
        if p_dropout is not None:
            model.add(Dropout(p_dropout))

    # Add output layer softmax activation function for classification
    model.add(Dense(n_classes, activation = 'softmax', name = 'output'))

    '''
    Add optimizer to the model and compile the model
    
    Optimizer used is Adam optimizer with learning rate from args or default value
    beta1 =0.9, beta2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False
    
    for the model compile part we use the loss and metrics from args or default values
    '''

    # Add optimizer to the model 
    opt = tf.keras.optimizers.Adam(learning_rate=args.lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    # Compile model
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
                  )

    # Return model
    return model