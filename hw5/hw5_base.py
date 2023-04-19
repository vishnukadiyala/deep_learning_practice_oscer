'''
Advanced Machine Learning, 2023
HW 5 Base Code

Author: Andrew H. Fagg (andrewhfagg@gmail.com)
Editor: Vishnu Kadiyala (vishnupk@ou.edu) 

Protien prediction for the pfamB dataset

'''

import sys
import argparse
import pickle
import pandas as pd
import py3nvml

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model

#############
# REMOVE THESE LINES if symbiotic_metrics, job_control, networks are in the same directory
tf_tools = "../../../../tf_tools/"
sys.path.append(tf_tools + "metrics")
sys.path.append(tf_tools + "experiment_control")
sys.path.append(tf_tools + "networks")
#############

# Provided
from symbiotic_metrics import *
from job_control import *
# from core50 import *
from pfam_loader import *

# You need to provide this yourself
from rcnn_model import *
import matplotlib.pyplot as plt
#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18

plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

#################################################################

def load_meta_data_set(fname = 'classes.pkl'):
    '''
    Load the metadata for a multi-class data set

    The returned data set is:
    - A python list of classes
    - Each class has a python list of object
    - Each object is described as using a DataFrame that contains file
       locations/names and class labels


    :param fname: Pickle file with the metadata

    :return: Dictionary containing class data
    '''

    obj = None
    
    with open(fname, "rb") as fp:
        obj = pickle.load(fp)

    assert obj is not None, "Meta-data loading error"
    
    print("Read %d classes\n"%obj['nclasses'])
    
    return obj


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='CNN', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')

    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu', help='Do not use the GPU')

    # High-level experiment configuration
    parser.add_argument('--exp_type', type=str, default=None, help="Experiment type")
    
    parser.add_argument('--label', type=str, default=None, help="Extra label to add to output files");
    parser.add_argument('--dataset', type=str, default='/home/fagg/datasets/pfam', help='Data set directory')
    parser.add_argument('--image_size', nargs=3, type=int, default=[3934], help="Size of input images (rows, cols, channels)")
    parser.add_argument('--meta_dataset', type=str, default='core50_df.pkl', help='Name of file containing the core 50 metadata')
    parser.add_argument('--Nfolds', type=int, default=5, help='Maximum number of folds')
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')

    # Specific experiment configuration
    parser.add_argument('--exp_index', type=int, default=None, help='Experiment index')
    parser.add_argument('--rotation', type=int, default=0, help='Cross-validation rotation')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--Ntraining', type=int, default=3, help='Number of training folds')
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--model', type=str, default='rnn', help="What model to use")
    # parser.add_argument('--unroll', action='store_true', help="Unroll the data")

    # Convolutional parameters
    parser.add_argument('--conv_size', nargs='+', type=int, default=[3,5], help='Convolution filter size per layer (sequence of ints)')
    parser.add_argument('--conv_nfilters', nargs='+', type=int, default=[10,15], help='Convolution filters per layer (sequence of ints)')
    parser.add_argument('--n_rnn', nargs='+', type=int, default=[3,5], help='RNN size per layer (sequence of ints)')
    parser.add_argument('--n_filters', nargs='+', type=int, default=[10,15], help='Convolution filters per layer (sequence of ints)')
    parser.add_argument('--pool', nargs='+', type=int, default=[2,2], help='Max pooling size (1=None)')
    parser.add_argument('--padding', type=str, default='valid', help='Padding type for convolutional layers')

    # Ebedding parameters
    parser.add_argument('--embeddings', type=int, default=25, help='Number of Embeddings')

    # Hidden unit parameters
    parser.add_argument('--hidden', nargs='+', type=int, default=[100, 5], help='Number of hidden units per layer (sequence of ints)')
    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--recurrent_dropout', type=float, default=None, help='Dropout rate for recurrent layers')
    parser.add_argument('--spatial_dropout', type=float, default=None, help='Dropout rate for convolutional layers')
    parser.add_argument('--L1_regularization', '--l1', type=float, default=None, help="L1 regularization parameter")
    parser.add_argument('--L2_regularization', '--l2', type=float, default=None, help="L2 regularization parameter")
    
    #Activation parameters
    parser.add_argument('--activation_dense', type=str, default='elu', help="Activation function for dense layers")
    parser.add_argument('--activation_rnn', type=str, default='tanh', help="Activation function for RNN layers")
    parser.add_argument('--activation_out', type=str, default='softmax', help="Activation function for output layer")

    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('--monitor', type=str, default="val_loss", help="Metric to monitor for early termination")

    # Training parameters
    parser.add_argument('--batch', type=int, default=10, help="Training set batch size")
    parser.add_argument('--prefetch', type=int, default=3, help="Number of batches to prefetch")
    parser.add_argument('--num_parallel_calls', type=int, default=4, help="Number of threads to use during batch construction")
    parser.add_argument('--cache', type=str, default=None, help="Cache (default: none; RAM: specify empty string; else specify file")
    parser.add_argument('--shuffle', type=int, default=0, help="Size of the shuffle buffer (0 = no shuffle")
    
    parser.add_argument('--generator_seed', type=int, default=42, help="Seed used for generator configuration")
    parser.add_argument('--repeat', action='store_true', help='Continually repeat training set')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help="Number of training batches per epoch (must use --repeat if you are using this)")

    # Image Augmentation: REVISIT TODO
    #parser.add_argument('--rotation_range', type=int, default=0, help="Image Generator: rotation range")
    #parser.add_argument('--width_shift_range', type=int, default=0, help="Image Generator: width shift range")
    #parser.add_argument('--height_shift_range', type=int, default=0, help="Image Generator: height shift range")
    #parser.add_argument('--shear_range', type=float, default=0.0, help="Image Generator: shift range")
    #parser.add_argument('--zoom_range', type=float, default=0.0, help="Image Generator: zoom range")
    #parser.add_argument('--horizontal_flip', action='store_true', help='Image Generator: horizontal flip')
    #parser.add_argument('--vertical_flip', action='store_true', help='Image Generator: vertical flip')

    # Post
    parser.add_argument('--save_model', action='store_true', default=False , help='Save a model file')
    parser.add_argument('--no-save_model', action='store_false', dest='save_model', help='Do not save a model file')
    
    return parser


def exp_type_to_hyperparameters(args):
    '''
    Translate the exp_type into a hyperparameter set

    This is trivial right now

    :param args: ArgumentParser
    :return: Hyperparameter set (in dictionary form)
    '''
    if args.exp_type is None:
        p = {'rotation': range(5)}
    else:
        assert False, "Unrecognized exp_type"

    return p


#################################################################
def check_args(args):
    assert (args.rotation >= 0 and args.rotation < args.Nfolds), "Rotation must be between 0 and Nfolds"
    assert (args.Ntraining >= 1 and args.Ntraining <= (args.Nfolds-1)), "Ntraining must be between 1 and Nfolds-2"
    assert (args.dropout is None or (args.dropout > 0.0 and args.dropout < 1)), "Dropout must be between 0 and 1"
    assert (args.spatial_dropout is None or (args.spatial_dropout > 0.0 and args.dropout < 1)), "Spatial dropout must be between 0 and 1"
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    assert (args.L1_regularization is None or (args.L1_regularization > 0.0 and args.L1_regularization < 1)), "L1_regularization must be between 0 and 1"
    assert (args.L2_regularization is None or (args.L2_regularization > 0.0 and args.L2_regularization < 1)), "L2_regularization must be between 0 and 1"
    assert (args.cpus_per_task is None or args.cpus_per_task > 1), "cpus_per_task must be positive or None"
    
def augment_args(args):
    '''
    Use the jobiterator to override the specified arguments based on the experiment index.

    Modifies the args

    :param args: arguments from ArgumentParser
    :return: A string representing the selection of parameters to be used in the file name
    '''
    
    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = exp_type_to_hyperparameters(args)

    # Check index number
    index = args.exp_index
    if(index is None):
        return ""
    
    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp_index, args)
 
    
#################################################################

def generate_fname(args, params_str):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.

    :param args: from argParse
    :params_str: String generated by the JobIterator
    '''
    # Hidden unit configuration
    hidden_str = '_'.join(str(x) for x in args.hidden)
    
    # Conv configuration
    conv_size_str = '_'.join(str(x) for x in args.conv_size)
    conv_filter_str = '_'.join(str(x) for x in args.conv_nfilters)
    pool_str = '_'.join(str(x) for x in args.pool)
    
    # Dropout
    if args.dropout is None:
        dropout_str = ''
    else:
        dropout_str = 'drop_%0.3f_'%(args.dropout)
        
    # Spatial Dropout
    if args.spatial_dropout is None:
        sdropout_str = ''
    else:
        sdropout_str = 'sdrop_%0.3f_'%(args.spatial_dropout)
        
    # L1 regularization
    if args.L1_regularization is None:
        regularizer_l1_str = ''
    else:
        regularizer_l1_str = 'L1_%0.6f_'%(args.L1_regularization)

    # L2 regularization
    if args.L2_regularization is None:
        regularizer_l2_str = ''
    else:
        regularizer_l2_str = 'L2_%0.6f_'%(args.L2_regularization)


    # Label
    if args.label is None:
        label_str = ""
    else:
        label_str = "%s_"%args.label
        
    # Experiment type
    if args.exp_type is None:
        experiment_type_str = ""
    else:
        experiment_type_str = "%s_"%args.exp_type

    # learning rate
    lrate_str = "LR_%0.6f_"%args.lrate
    
    # Put it all together, including #of training folds and the experiment rotation
    return "%s/image_%s%sCsize_%s_Cfilters_%s_Pool_%s_Pad_%s_hidden_%s_%s%s%s%s%sntrain_%02d_rot_%02d"%(args.results_path,
                                                                                           experiment_type_str,
                                                                                           label_str,
                                                                                           conv_size_str,
                                                                                           conv_filter_str,
                                                                                           pool_str,
                                                                                           args.padding,
                                                                                           hidden_str, 
                                                                                           dropout_str,
                                                                                           sdropout_str,
                                                                                           regularizer_l1_str,
                                                                                           regularizer_l2_str,
                                                                                           lrate_str,
                                                                                           args.Ntraining,
                                                                                           args.rotation)

#################################################################

def build_regularization_kernel(lambda_l1, lambda_l2):
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
    
    
    return kernel 
#################################################################
def execute_exp(args=None, multi_gpus=False):
    '''
    Perform the training and evaluation for a single model
    
    :param args: Argparse arguments
    '''
    
    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
        
    print(args.exp_index)
    
    # Override arguments if we are using exp_index
    args_str = augment_args(args)

    # Scale the batch size with the number of GPUs
    if multi_gpus > 1:
        args.batch = args.batch*multi_gpus

    ####################################################
    # Create the TF datasets for training, validation, testing
    # HW3
    #ds_train, ds_validation, ds_testing, n_classes = load_data_set(args)
    # HW4
    # ds_train, ds_validation, ds_testing, n_classes = load_data_set_by_folds(args, objects = list(range(10)))
    
    #HW5 
    data_out = load_rotation(basedir = args.dataset, rotation=args.rotation)
    ds_train, ds_validation, ds_testing = create_tf_datasets(data_out, batch=args.batch, prefetch=args.prefetch, repeat=args.repeat, shuffle=args.shuffle)
    n_classes = data_out['n_classes']
    len_max = data_out['len_max']
    rotation = data_out['rotation']
    n_tokens = data_out['n_tokens']
    print(n_tokens)
    print(len_max)
    print(n_classes)
    
    del data_out
    # exit()
    
    ####################################################
  

    # Network config
    # NOTE: this is very specific to our implementation of create_cnn_classifier_network()
    #   List comprehension and zip all in one place (ugly, but effective).
    #   Feel free to organize this differently
    # dense_layers = [{'units': i} for i in args.hidden]
    
    # conv_layers = [{'filters': f, 'kernel_size': (s,s), 'pool_size': (p,p), 'strides': (p,p)} if p > 1
    #                else {'filters': f, 'kernel_size': (s,s), 'pool_size': None, 'strides': None}
    #                for s, f, p, in zip(args.conv_size, args.conv_nfilters, args.pool)]
    
    # rnn_layers = [{'n_rnn': i} for i in args.n_rnn]
    

    kernel = build_regularization_kernel(args.L1_regularization, args.L2_regularization)

    if multi_gpus > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():

            if args.model == 'rnn':
                print("Running RNN model")
            # Build network: you must provide your own implementation
                model = create_srnn_classifier_network(n_tokens = n_tokens,
                                            len_max = len_max,
                                            n_embeddings = args.embeddings,
                                            n_rnn = args.n_rnn,
                                            activation = args.activation_rnn,
                                            hidden = args.hidden,
                                            activation_hidden = args.activation_dense,
                                            n_outputs = n_classes,
                                            activation_output = args.activation_out,
                                            dropout = args.dropout,
                                            recurrent_dropout = args.recurrent_dropout,
                                            lrate = args.lrate,
                                            lamda_regularization = kernel,
                                            binding_threshold = 0.42,
                                            loss = 'sparse_categorical_crossentropy',
                                            metrics = ['sparse_categorical_accuracy'],)
            elif args.model == 'cnn':
                print("Running CNN model")
                pass 
                
            
            
            else: 
                print("Model is not defined")
                exit()
            
    else:
        if args.model == 'rnn':
            
            if args.pool is not None:
                print("Running RNN model with average pooling")
            else:   
                print("Running RNN model")
            
            model = create_srnn_classifier_network(n_tokens = n_tokens,
                                          len_max = len_max,
                                          n_embeddings = args.embeddings,
                                          n_rnn = args.n_rnn,
                                          activation = args.activation_rnn,
                                          hidden = args.hidden,
                                          activation_hidden = args.activation_dense,
                                          n_outputs = n_classes,
                                          activation_output = args.activation_out,
                                          avg_pooling=args.pool,
                                          dropout = args.dropout,
                                          recurrent_dropout = args.recurrent_dropout,
                                          lrate = args.lrate,
                                          unroll=False,
                                          lamda_regularization = kernel,
                                          binding_threshold = 0.42,
                                          loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                          metrics = ['sparse_categorical_accuracy'],
                                          )
        elif args.model == 'cnn':
            print("Running CNN model")

            model = cnn_classifier(
                                    n_tokens = n_tokens,
                                    len_max = len_max,
                                    n_embeddings = args.embeddings,
                                    n_cnn = args.conv_size,
                                    n_filters = args.n_filters,
                                    hidden = args.hidden,
                                    activation_hidden = args.activation_dense,
                                    n_outputs = n_classes,
                                    avg_pooling=args.pool,
                                    activation_output = args.activation_out,
                                    dropout = args.dropout,
                                    spatial_dropout = args.spatial_dropout,
                                    lrate = args.lrate,
                                    lamda_regularization = kernel,
                                    loss = 'sparse_categorical_crossentropy',
                                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
            
        
        elif args.model == 'gru':
            print("Running GRU model")
            
            model = create_gru_network(
                                        n_tokens = n_tokens,
                                        len_max = len_max,
                                        n_embeddings = args.embeddings,
                                        n_rnn = args.n_rnn,
                                        n_cnn = args.n_filters,
                                        n_filters = args.n_filters,
                                        activation = args.activation_rnn,
                                        hidden = args.hidden,
                                        avg_pooling = args.pool,
                                        activation_hidden = args.activation_dense,
                                        n_outputs = n_classes,
                                        activation_output = args.activation_out,
                                        dropout = args.dropout,
                                        recurrent_dropout = args.recurrent_dropout,
                                        lrate = args.lrate,
                                        lambda_regularization = kernel,
                                        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                                        )
        
        elif args.model == 'lstm':
            print("Running LSTM model")
            
            model = create_lstm_network(
                                        n_tokens = n_tokens,
                                        len_max = len_max,
                                        n_embeddings = args.embeddings,
                                        n_rnn = args.n_rnn,
                                        n_cnn = args.n_filters,
                                        n_filters = args.n_filters,
                                        activation = args.activation_rnn,
                                        hidden = args.hidden,
                                        conv_size= args.pool,
                                        activation_hidden = args.activation_dense,
                                        n_outputs = n_classes,
                                        activation_output = args.activation_out,
                                        dropout = args.dropout,
                                        recurrent_dropout = args.recurrent_dropout,
                                        lrate = args.lrate,
                                        lambda_regularization = kernel,
                                        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                                        )
        else: 
            print("Model is not defined")
            exit() 
    # Report model structure if verbosity is turned on
    if args.verbose >= 1:
        print(model.summary())
        print("Model")

    print(args)

    # Output file base and pkl file
    fbase = generate_fname(args, args_str)
    fname_out = "%s_results.pkl"%fbase

    # Plot the model
    plot_model(model, to_file='%s_model_plot.png'%fbase, show_shapes=True, show_layer_names=True)

    # Perform the experiment?
    if(args.nogo):
        # No!
        print("NO GO")
        print(fbase)
        return

    # Check if output file already exists
    if os.path.exists(fname_out):
            # Results file does exist: exit
            print("File %s already exists"%fname_out)
            return
            
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True,
                                                      min_delta=args.min_delta, monitor=args.monitor)

    # Learn
    #  steps_per_epoch: how many batches from the training set do we use for training in one epoch?
    #          Note that if you use this, then you must repeat the training set
    #  validation_steps=None means that ALL validation samples will be used
    history = model.fit(ds_train,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        use_multiprocessing=True, 
                        verbose=args.verbose>=2,
                        validation_data=ds_validation,
                        callbacks=[early_stopping_cb])


    
    # Generate results data
    results = {}
    results['args'] = args
    results['predict_validation'] = model.predict(ds_validation)
    results['predict_validation_eval'] = model.evaluate(ds_validation)
    
    if ds_testing is not None:
        results['predict_testing'] = model.predict(ds_testing)
        results['predict_testing_eval'] = model.evaluate(ds_testing)
        
    results['predict_training'] = model.predict(ds_train)
    results['predict_training_eval'] = model.evaluate(ds_train)
    results['history'] = history.history

    # Save results
    fbase = generate_fname(args, args_str)
    results['fname_base'] = fbase
    with open("%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    if args.save_model:
        model.save("%s_model"%(fbase))
    
    print(fbase)
    
    del results, model, history, ds_train, ds_validation, ds_testing 
    args.exp_index += 1
    if args.exp_index < 5:
        execute_exp(args)
    
    return None


def check_completeness(args):
    '''
    Check the completeness of a Cartesian product run.

    All other args should be the same as if you executed your batch, however, the '--check' flag has been set

    Prints a report of the missing runs, including both the exp_index and the name of the missing results file

    :param args: ArgumentParser

    '''
    
    # Get the corresponding hyperparameters
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)

    print("Total jobs: %d"%ji.get_njobs())

    print("MISSING RUNS:")

    indices = []
    # Iterate over all possible jobs
    for i in range(ji.get_njobs()):
        params_str = ji.set_attributes_by_index(i, args)
        # Compute output file name base
        fbase = generate_fname(args, params_str)
    
        # Output pickle file name
        fname_out = "%s_results.pkl"%(fbase)

        if not os.path.exists(fname_out):
            # Results file does not exist: report it
            print("%3d\t%s"%(i, fname_out))
            indices.append(i)

    # Give the list of indices that can be inserted into the --array line of the batch file
    print("Missing indices (%d): %s"%(len(indices),','.join(str(x) for x in indices)))

    
#################################################################
if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    
    # Turn off GPU?
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
    # GPU check
    physical_devices = tf.config.list_physical_devices('GPU') 
    n_physical_devices = len(physical_devices)
    print(physical_devices)
    if(n_physical_devices > 0):
        py3nvml.grab_gpus(num_gpus=n_physical_devices, gpu_select=range(n_physical_devices))
        # for device in physical_devices:
            # tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n'%n_physical_devices)
    else:
        print('NO GPU')


    if(args.check):
        # Just check to see if all experiments have been executed
        check_completeness(args)
    else:
        # Execute the experiment

        # Set number of threads, if it is specified
        if args.cpus_per_task is not None:
            tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
            tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)

        execute_exp(args, multi_gpus=n_physical_devices)
        
