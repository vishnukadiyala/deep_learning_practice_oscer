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
    lrate=0.001,
    n_classes = 10,
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
    padding = 'same',
    flatten = True,
    args = args):

    # Create model
    model = tf.keras.Sequential()
    
    # Add convolutional layers (Input layer)
    model.add(Conv2D(32, (3, 3), padding=padding, input_shape=(image_size[0],image_size[1], nchannels)))
    
    # Add additional convolutional layers
    for i, n in enumerate(conv_layers):
        model.add(Conv2D(n['filters'], n['kernel_size'],padding = padding, name = 'conv_{}'.format(i)))
        
        # Add Spatial Dropout if exists
        if p_spatial_dropout is not None: 
            model.add(SpatialDropout2D(p_spatial_dropout))
            
        #Add Max Pooling if exists
        if n['pool_size'] is not None:    
            model.add(MaxPooling2D(pool_size = n['pool_size'], strides = n['strides'], name = 'pool_{}'.format(i+1)))
    
    # Global Max Pooling Layer to find the most important features
    model.add(GlobalMaxPooling2D())
    
    # Add dense layers
    
    #model.add(Flatten())
    # Dense layers
    for i,n in enumerate(dense_layers):
        # Check if regularization exists, then add a dense layer with regularization
        if lambda_l2 is not None:
            model.add(Dense(n['units'], activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(lambda_l2), name = 'dense_{}'.format(i+1)))
        
        # else add a dense layer without regularization
        else: 
            model.add(Dense(n['units'], activation = 'relu', name = 'dense_{}'.format(i+1)))

        # Add dropout if exists
        if p_dropout is not None:
            model.add(Dropout(p_dropout))

    
    # Add output layer softmax activation function for classification
    model.add(Dense(n_classes, activation = 'softmax', name = 'output'))


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