'''
Deep Learning Demo: XOR

Command line version

Andrew H. Fagg (andrewhfagg@gmail.com)
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time
import re

import argparse
import pickle

# Tensorflow 2.x way of doing things
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential

#################################################################
# Default plotting parameters
FONTSIZE = 14
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = FONTSIZE

#################################################################
def build_model(n_inputs, n_hidden, n_output, activation='elu', lrate=0.001):
    '''
    Construct a network with one hidden layer
    - Adam optimizer
    - MSE loss
    
    :param n_inputs: Number of input dimensions
    :param n_hidden: Number of units in the hidden layer
    :param n_output: Number of ouptut dimensions
    :param activation: Activation function to be used for hidden and output units
    :param lrate: Learning rate for Adam Optimizer
    '''
    model = Sequential();
    model.add(InputLayer(input_shape=(n_inputs,)))
    #for i, n in enumerate(n_hidden):
    model.add(Dense(n_hidden, use_bias=True, name="hidden_1", activation=activation))

    # model.add(Dense(n_hidden, use_bias=True, name="hidden_2", activation=activation))
   
    model.add(Dense(n_output, use_bias=True, name="output", activation=activation))

    
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    
    # Bind the optimizer and the loss function to the model
    model.compile(loss='mse', optimizer=opt)
    
    # Generate an ASCII representation of the architecture
    print(model.summary())
    
    return model
def args2string(args):
    '''
    Translate the current set of arguments
    
    :param args: Command line arguments
    '''
    return "exp_%02d_hidden_%02d"%(args.exp, args.hidden)
    
    
########################################################
def execute_exp(args):
    '''
    Execute a single instance of an experiment.  The details are specified in the args object
    
    :param args: Command line arguments
    '''

    ##############################
    # Run the experiment
    # Create training set: XOR

    ins = np.array(foo['ins'])
    outs = np.array(foo['outs'])

    #print(ins[1].shape)
    
    #Build Model
    model = build_model(ins.shape[1], args.hidden, outs.shape[1], activation = args.activation)

    # Callbacks
    
    early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10000,restore_best_weights = True, min_delta = 0.0001)

    # Describe arguments
    argstring = args2string(args)
    print("EXPERIMENT: %s"%argstring)
    
    # Only execute if we are 'going'
    if not args.nogo:
        # Training
        print("Training...")
        
        # Note: faking validation data set
        history = model.fit( x = ins,
                            y = outs,
                            verbose = False,
                            validation_data = (ins,outs),
                            epochs = args.epochs,
                            callbacks = [early_stopping_cb]
            )
        
        # print(history.history['loss'])

        print("Done Training")

        print("Predicting... ")

        predictions = model.predict(ins)
        pred_error = np.abs(predictions - outs)
        
        # Save the training history
        fp = open("results/hw00_results_%s.pkl" %(argstring), "wb")
        pickle.dump(history.history, fp)
        pickle.dump(args,fp)
        # Save Prediction errors in the same pickle file
        pickle.dump(pred_error, fp)             
        
        fp.close()

def display_learning_curve():
    '''
    Display the learning curve that is stored in fname
    
    :param fname: Results file to load and dipslay
    
    '''
    
    # Load the history file and display it
    #fpTODO
    #TODO
    argstring = args2string(args)
    fp = open("results/hw0_results_%s.pkl" %(argstring), "rb")
    history = pickle.load(fp)
    fp.close()
    
    # Display
    plt.plot(history['loss'])
    plt.ylabel('MSE')
    plt.xlabel('epochs')
    plt.savefig("exp%s.png"%(argstring))

def display_learning_curve_set(dir, base):
    '''
    Plot the learning curves for a set of results
    
    :param base: Directory containing a set of results files
    '''
    # Find the list of files in the local directory that match base_[\d]+.pkl
    abs_error = np.array([])
    files = [f for f in os.listdir(dir) if re.match(r'%s.+.pkl'%(base), f)]
    files.sort()
    
    # Load back data from the pickle files
    for f in files:
        with open("%s/%s"%(dir,f), "rb") as fp:
            history = pickle.load(fp)
            plt.plot(history['loss'])
            args = pickle.load(fp)
            pred_error = pickle.load(fp)
            # add all data to absolute error
            abs_error = np.append(abs_error, pred_error)

    # Plot Learning Curves 
    plt.ylabel('MSE')
    plt.xlabel('epochs')
    plt.legend(files)
    plt.savefig('Training.png')
    plt.clf()

    # Plot histogram
    plt.ylabel('Frequency')
    plt.xlabel('Absolute error')
    plt.hist(abs_error, 50)
    plt.savefig('hist.png')



def create_parser():
    '''
    Create a command line parser for the XOR experiment
    '''
    parser = argparse.ArgumentParser(description='XOR Learner')
    
    parser.add_argument('--exp', type=int, default=0, help='Experiment index')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--hidden', type=int, default=2, help='Number of Hidden Units')
    #parser.add_argument('--hidden', nargs='+', type=int, default=[5], help = 'Number of hidden units per layer')
    parser.add_argument('--lrate', type = float, default=None, help='Learning rate')
    parser.add_argument('--activation', type = str, default = 'elu', help = 'Activation Function')
    
    # BOOLEAN SWITCH
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')

    #parser.add_argument('--v')
    
    
    return parser

'''
This next bit of code is executed only if this python file itself is executed
(if it is imported into another file, then the code below is not executed)
'''
if __name__ == "__main__":
    # Parse the command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Turn off GPU?
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Read in the data
    fp = open("hw0_dataset.pkl", "rb")
    foo = pickle.load(fp)
    fp.close()
    
    # GPU check
    physical_devices = tf.config.list_physical_devices('GPU') 
    n_physical_devices = len(physical_devices)
    if(n_physical_devices > 0):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('We have %d GPUs\n'%n_physical_devices)
    else:
        print('NO GPU')


    # Do the work
    execute_exp(args)

    # # Show plots 
    # Set experiment number 100 with --nogo to create plots 
    # Previously set to args.exp == 9: and it created plots
    # before the execution as all the experiments ran simultaneously on OSCER
    if(args.exp == 100):
        path = os.getcwd() + str("/results")
        base = str("hw0")
        display_learning_curve_set(path, base)

