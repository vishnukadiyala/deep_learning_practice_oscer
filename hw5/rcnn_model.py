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
from keras.models import Sequential

#   Create parser
args = argparse.ArgumentParser()



def create_srnn_classifier_network(
                                    n_tokens = 1000,
                                    len_max = 100,
                                    n_embeddings = 25,
                                    n_rnn = [100,100],
                                    activation = 'tanh',
                                    hidden = [100,100],
                                    activation_hidden = 'elu',
                                    n_outputs = 10,
                                    activation_output = 'softmax',
                                    dropout = None,
                                    recurrent_dropout = None,
                                    lrate = 0.001,
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
                            name = 'rnn_layer_{}'.format(i+1)
                            ))

    model.add(SimpleRNN(
                        units = n_rnn[-1],
                        activation = activation,
                        return_sequences = False,
                        dropout = dropout,
                        recurrent_dropout = recurrent_dropout,
                        kernel_regularizer = lamda_regularization,
                        recurrent_regularizer = lamda_regularization,
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
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    
    # Compile model
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
                  )
    
    return model
