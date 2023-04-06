# required packages: tensorflow, keras, pandas, numpy, matplotlib, sklearn
import tensorflow as tf 
import sys
import argparse
import pickle
import pandas as pd
import numpy as np


# Model building packages
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, SimpleRNN
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from keras.layers import GlobalMaxPooling2D
from keras.layers import SpatialDropout2D
from keras.layers import InputLayer
from keras.layers import Embedding
from keras import Input
from keras.layers import Concatenate
from keras import Model

#   Create parser
args = argparse.ArgumentParser()


def inception_module(x, filters, name=None):
    
    print('Inception Module Added')
    # Branch A 
    conv_A = Conv2D(filters, (1, 1), strides = (2,2), padding='same', activation='elu', name=name + '_convA')(x)
    
    # Branch B
    conv_B = Conv2D(filters, (1, 1), strides = (1 , 1), padding='same', activation='elu', name=name + '_convB_1')(x)
    conv_B = Conv2D(filters, (3, 3), strides = (2 , 2), padding='same', activation='elu', name=name + '_convB_2')(conv_B)
    
    # Branch C 
    ## Added 2 3x3 conv layers instead of 1 5x5 conv layer. Based on the paper inception v2, Makes it computationally efficient.
    conv_C = Conv2D(filters, (1, 1), strides = (1 , 1), padding='same', activation='elu', name=name + '_convC_1')(x)
    conv_C = Conv2D(filters, (3, 3), strides = (1 , 1), padding='same', activation='elu', name=name + '_convC_2')(conv_C)
    conv_C = Conv2D(filters, (3, 3), strides = (2 , 2), padding='same', activation='elu', name=name + '_convC_3')(conv_C)
    
    # Branch D 
    conv_D = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=name + '_ConvD_1')(x)
    conv_D = Conv2D(filters, (3, 3), strides = (2 , 2), padding='same', activation='elu', name=name + '_convD_2')(conv_D)
    
    # Concatenate all the branches
    x_concat = Concatenate(name=name+'_concat')([conv_A, conv_B, conv_C, conv_D])
    
    # Return concatenated layer
    return x_concat

def create_srnn_classifier_network(
    data_size = None,
    conv_layers = None,
    dense_layers = None,
    p_dropout = None,
    lambda_l2 = None,
    lambda_l1 = None,
    lrate=0.001,
    n_classes = 10,
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
    padding = 'same',
    flatten = True,
    args = args
):
    
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
    Model Building Part 
    '''
    input_tensor = Input(input_length = data_size)
    
    embedding_tensor = Embedding(input_dim = data_size, output_dim = int (data_size/2))(input_tensor)
    
    rnn_tensor = SimpleRNN(units = 100,
                              activation='tanh',
                              dropout=p_dropout,
                              kernel_regularizer=kernel,
                              unroll=args.gpu,
                              )(embedding_tensor)
    
    output_tensor = Dense(n_classes, activation='softmax')(rnn_tensor)
    '''
    #Add optimizer to the model and compile the model
    
    Optimizer used is Adam optimizer with learning rate from args or default value
    beta1 =0.9, beta2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False
    
    for the model compile part we use the loss and metrics from args or default values
    '''
    # Create model
    model = Model(inputs = input_tensor, outputs = output_tensor)
    # Add optimizer to the model 
    opt = tf.keras.optimizers.Adam(learning_rate=args.lrate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    # Compile model
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
                  )
    
    return model



# Model Building function with default parameters
def create_cnn_classifier_network(
    data_size = None,
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
    #model = tf.keras.Sequential()
    
    # Add Input Layer with image size and number of channels, also add Spatial Dropout if exists
    #x = InputLayer(input_shape=(image_size[0],image_size[1], nchannels))
    input_tensor = Input(shape=(1, data_size,))
    
    if p_spatial_dropout is not None: 
        x = SpatialDropout2D(p_spatial_dropout)(input_tensor)
    else:
        x = input_tensor
    # Add additional convolutional layers based on conv_layers input file
    for i, n in enumerate(conv_layers):
        if n['kernel_size'] == (25,25):
            x = inception_module(x, n['filters'], name = 'inception_{}'.format(i))
            
            
            # Add Spatial Dropout if exists after each inception module
            if p_spatial_dropout is not None: 
                x = SpatialDropout2D(p_spatial_dropout)(x)
                
        else:
            x = Conv1D(n['filters'],kernel_size=1 ,padding = padding, activation = 'elu', name = 'conv_{}'.format(i))(x)
            
             # Add Spatial Dropout if exists
            if p_spatial_dropout is not None: 
                x = SpatialDropout2D(p_spatial_dropout)(x)
                
            #Add Max Pooling if exists
            # We will add Max Pooling after each convolutional layer only if pool_size > 1. Used this condition to give an input from text file
            if n['pool_size'] is not None and n['pool_size'][0] > 1:    
                x = MaxPooling1D(pool_size = 1, strides = 1, name = 'pool_{}'.format(i+1))(x)
        
    # Global Max Pooling Layer to find the most important features
    #x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    # Add dense layers
    
    for i,n in enumerate(dense_layers):
        
        #add dense layer with kernel regularization
        x = Dense(n['units'], activation = 'elu', kernel_regularizer = kernel, name = 'dense_{}'.format(i+1))(x)
        
        # Add dropout if exists
        if p_dropout is not None:
            x = Dropout(p_dropout)(x)

    # Add output layer softmax activation function for classification
    output_tensor = Dense(n_classes, activation = 'softmax', name = 'output')(x)

    '''
    Add optimizer to the model and compile the model
    
    Optimizer used is Adam optimizer with learning rate from args or default value
    beta1 =0.9, beta2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False
    
    for the model compile part we use the loss and metrics from args or default values
    '''
    # Create model
    model = Model(inputs = input_tensor, outputs = output_tensor)
    # Add optimizer to the model 
    opt = tf.keras.optimizers.Adam(learning_rate=args.lrate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    # Compile model
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
                  )

    # Return model
    return model

