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
from keras.layers import GlobalMaxPooling2D, AveragePooling3D, AveragePooling1D, AveragePooling2D
from keras.layers import SpatialDropout2D
from keras.layers import InputLayer
from keras.layers import Embedding
from keras import Input
from keras.layers import Concatenate
from keras import Model
from keras.models import Sequential
from keras.layers import GRU, LSTM

#   Create parser
args = argparse.ArgumentParser()

def create_gru_network(
    n_tokens = 1000,
    len_max = 100,
    n_embeddings = 25,
    n_rnn = [100,100],
    n_cnn = [100,100],
    n_filters = 100,
    activation = 'tanh',
    hidden = [100,100],
    avg_pooling = None,
    activation_hidden = 'elu',
    n_outputs = 10,
    activation_output = 'softmax',
    dropout = None,
    recurrent_dropout = None,
    lrate = 0.001,
    lambda_regularization = None,
    loss = 'sparse_categorical_crossentropy',
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    ):
    '''
    This function creates a GRU network with the following parameters:
    
    Embedding Layer:
        n_tokens = 1000
        n_embeddings = 25
        len_max = 100
    
    Convolutional Layers:
        elu activation
        average pooling
    
    RNN Layers:
        tanh activation
    
    Dense Layers:
        elu activation
        softmax activation for the last layer 
    
    Loss Function:
        sparse_categorical_crossentropy
    
    metrics:
        SparseCategoricalAccuracy    
     
    '''
    
    model = Sequential()
    
    # Embedding Layer
    model.add(Embedding(n_tokens, n_embeddings, input_length = len_max))
    
    # Convolutional Layers
    
    for i,n in enumerate(n_cnn):
        
        model.add(Conv1D(
                        filters = n,
                        kernel_size = 3,
                        activation = activation_hidden,
                        padding = 'same',
                        name = 'conv_layer_{}'.format(i+1)
                        ))
        
        if avg_pooling is not None:
            model.add(AveragePooling1D(
                        pool_size = avg_pooling[i],
                        name = 'avg_pooling_layer_{}'.format(i+1)
                        ))
        
        if dropout is not None:
            model.add(Dropout(dropout))

    for i,n in enumerate(n_rnn[:-1]):
        
        model.add(GRU(
                        units = n,
                        activation = activation,
                        return_sequences = True,
                        dropout = dropout,
                        recurrent_dropout = recurrent_dropout,
                        kernel_regularizer = lambda_regularization,
                        recurrent_regularizer = lambda_regularization,
                        unroll=True,
                        name = 'gru_layer_{}'.format(i+1)
                        ))
    model.add(GRU(
                    units = n_rnn[-1],
                    activation = activation,
                    return_sequences = False,
                    dropout = dropout,
                    recurrent_dropout = recurrent_dropout,
                    kernel_regularizer = lambda_regularization,
                    recurrent_regularizer = lambda_regularization,
                    unroll=True,
                    name = 'gru_layer_last'
                    ))
    
    for i,n in enumerate(hidden):
        
        model.add(Dense(
                        units = n,
                        activation = activation_hidden,
                        kernel_regularizer = lambda_regularization,
                        use_bias = True,
                        bias_initializer = 'zeros',
                        kernel_initializer = 'truncated_normal',
                        name = 'dense_layer_{}'.format(i+1)
                        ))
        if dropout is not None:
            model.add(Dropout(dropout))
    
    model.add(Dense(n_outputs, activation = activation_output))
    
    
    '''
    #Add optimizer to the model and compile the model
    
    Optimizer used is Adam optimizer with learning rate from args or default value
    
    for the model compile part we use the loss and metrics from args or default values
    '''
    # Add optimizer to the model 
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=True, clipvalue=0.5,beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adam' )
    
    # Compile model
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
                  )
    
    return model

def create_srnn_classifier_network(
                                    n_tokens = 1000,
                                    len_max = 100,
                                    n_embeddings = 25,
                                    n_rnn = [100,100],
                                    activation = 'tanh',
                                    hidden = [100,100],
                                    avg_pooling = None,
                                    activation_hidden = 'elu',
                                    n_outputs = 10,
                                    activation_output = 'softmax',
                                    dropout = None,
                                    recurrent_dropout = None,
                                    lrate = 0.001,
                                    unroll = False,
                                    lamda_regularization = None,
                                    binding_threshold = 0.42,
                                    loss = 'sparse_categorical_crossentropy',
                                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],):
    

    ''' 
    Model Building Part 
    '''
    model = Sequential()
    
    # ADD Embedding layer
    
    model.add(Embedding(n_tokens, n_embeddings, input_length = len_max))
    
    # Add RNN layers 
    
    for i,n in enumerate(n_rnn[:-1]):
        
        model.add(SimpleRNN(
                            units = n, 
                            activation = activation, 
                            return_sequences = True, 
                            dropout = dropout, 
                            recurrent_dropout = recurrent_dropout,
                            kernel_regularizer = lamda_regularization,
                            recurrent_regularizer = lamda_regularization,
                            unroll=True,
                            name = 'rnn_layer_{}'.format(i+1)
                            ))
        if avg_pooling is not None:
           model.add(AveragePooling1D(pool_size=avg_pooling[i], strides=None, padding='valid', name = 'pooling_layer_{}'.format(i+1)))
        
    

    model.add(SimpleRNN(
                        units = n_rnn[-1],
                        activation = activation,
                        return_sequences = False,
                        dropout = dropout,
                        recurrent_dropout = recurrent_dropout,
                        kernel_regularizer = lamda_regularization,
                        recurrent_regularizer = lamda_regularization,
                        unroll=True,
                        name = 'rnn_layer_last' ))

    
    
    # Add dense Layers 
    for i,n in enumerate(hidden):
        
        model.add(Dense(
                        units = n,
                        activation = activation_hidden,
                        kernel_regularizer = lamda_regularization,
                        use_bias = True,
                        bias_initializer = 'zeros',
                        kernel_initializer = 'truncated_normal',
                        name = 'dense_layer_{}'.format(i+1)
                        ))
        if dropout is not None:
            model.add(Dropout(dropout))
    
    model.add(Dense(n_outputs,
                    activation = activation_output,
                    use_bias = True,
                    bias_initializer = 'zeros',
                    kernel_regularizer = lamda_regularization,
                    kernel_initializer = 'truncated_normal',
                    name = 'output_layer'
                    ))
        
    '''
    #Add optimizer to the model and compile the model
    
    Optimizer used is Adam optimizer with learning rate from args or default value
    
    for the model compile part we use the loss and metrics from args or default values
    '''
    # Create model
    # model = Model(inputs = input_tensor, outputs = output_tensor)
    # Add optimizer to the model 
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=True, clipvalue=0.5,beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adam' )
    
    # Compile model
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
                  )
    
    return model

def cnn_classifier(
                                    n_tokens = 1000,
                                    len_max = 100,
                                    n_embeddings = 25,
                                    n_cnn = [100,100],
                                    n_filters = [100,100],
                                    hidden = [100,100],
                                    avg_pooling = None,
                                    activation_hidden = 'elu',
                                    n_outputs = 10,
                                    activation_output = 'softmax',
                                    dropout = None,
                                    spatial_dropout = None,
                                    lrate = 0.001,
                                    lamda_regularization = None,
                                    loss = 'sparse_categorical_crossentropy',
                                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                                    ):
    '''
    CNN model Building Part 

    '''
    
    model = Sequential()
    
    # ADD Embedding layer
    model.add(Embedding(n_tokens, n_embeddings, input_length = len_max))
    
    # Add CNN layers
    for i,n in enumerate(n_cnn):
        model.add(Conv1D(n_filters[i], n ,padding = 'valid', activation = activation_hidden, name = 'conv_{}'.format(i)))
        
       # if avg_pooling[i] is not None :
       #     model.add(AveragePooling1D(pool_size=4, strides=2, padding='valid', name = 'pooling_layer_{}'.format(i+1)))
        
        if spatial_dropout is not None:
            model.add(Dropout(spatial_dropout))
    
    # Add Flatten layer
    
    model.add(Flatten())

    # Add dense Layers

    for i,n in enumerate(hidden):
        
        #add dense layer with kernel regularization
        model.add(Dense(n, activation = activation_hidden, kernel_regularizer = lamda_regularization, name = 'dense_{}'.format(i+1)))
        
        # Add dropout if exists
        if dropout is not None:
            model.add(Dropout(dropout))

    # Add output layer softmax activation function for classification
    model.add(Dense(n_outputs, activation = 'softmax', name = 'output'))

            
    
    '''
    Add optimizer to the model and compile the model
    
    '''
    
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=True, clipvalue=0.5,beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adam' )
    
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    return model

def create_lstm_network(
                        n_tokens = 1000,
                        len_max = 100,
                        n_embeddings = 25,
                        n_rnn = [100,100],
                        n_cnn = None,
                        n_filters = [100,100],
                        activation = 'tanh',
                        hidden = [100,100],
                        conv_size = None,
                        activation_hidden = 'elu',
                        n_outputs = 10,
                        activation_output = 'softmax',
                        dropout = None,
                        recurrent_dropout = None,
                        lrate = 0.001,
                        lambda_regularization = None,
                        loss = 'sparse_categorical_crossentropy',
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                        ):
    '''
    
    Create an LSTM network
    '''
    
    model = Sequential()
    
    model.add(Embedding(n_tokens, n_embeddings, input_length = len_max))
    
    if n_filters is not None:
        for i,n in enumerate(n_filters):
            model.add(Conv1D(n, 
                             conv_size[i],
                             padding = 'valid', 
                             activation = activation_hidden, 
                             name = 'conv_{}'.format(i)))
            
            if dropout is not None:
                model.add(Dropout(dropout))
    
    for i,n in enumerate(n_rnn[:-1]):
        model.add(LSTM(n, 
                       activation = activation, 
                       return_sequences = True, 
                       dropout = dropout, 
                       recurrent_dropout = recurrent_dropout, 
                       kernel_regularizer = lambda_regularization, 
                       recurrent_regularizer = lambda_regularization, 
                       unroll=False,
                       name = 'lstm_layer_{}'.format(i+1)))
    
    model.add(LSTM(n_rnn[-1], 
                   activation = activation, 
                   return_sequences = False, 
                   dropout = dropout, 
                   recurrent_dropout = recurrent_dropout, 
                   kernel_regularizer = lambda_regularization, 
                   recurrent_regularizer = lambda_regularization, 
                   unroll=False, 
                   name = 'lstm_layer_last'))
    
    for i,n in enumerate(hidden):
        
        model.add(Dense(
                        units = n,
                        activation = activation_hidden,
                        kernel_regularizer = lambda_regularization,
                        use_bias = True,
                        bias_initializer = 'zeros',
                        kernel_initializer = 'truncated_normal',
                        name = 'dense_layer_{}'.format(i+1)
                        ))
        if dropout is not None:
            model.add(Dropout(dropout))
            
    model.add(Dense(n_outputs,
                    activation = activation_output,
                    name = 'output_layer'))
    
    '''
    Add optimizer to the model and compile the model
    
    '''
    
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=True, clipvalue=0.5,beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adam' )
    
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    return model    
    
    