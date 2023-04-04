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
from cnn_classifier import *
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

    # Convolutional parameters
    parser.add_argument('--conv_size', nargs='+', type=int, default=[3,5], help='Convolution filter size per layer (sequence of ints)')
    parser.add_argument('--conv_nfilters', nargs='+', type=int, default=[10,15], help='Convolution filters per layer (sequence of ints)')
    parser.add_argument('--pool', nargs='+', type=int, default=[2,2], help='Max pooling size (1=None)')
    parser.add_argument('--padding', type=str, default='valid', help='Padding type for convolutional layers')

    # Hidden unit parameters
    parser.add_argument('--hidden', nargs='+', type=int, default=[100, 5], help='Number of hidden units per layer (sequence of ints)')

    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--spatial_dropout', type=float, default=None, help='Dropout rate for convolutional layers')
    parser.add_argument('--L1_regularization', '--l1', type=float, default=None, help="L1 regularization parameter")
    parser.add_argument('--L2_regularization', '--l2', type=float, default=None, help="L2 regularization parameter")

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
def load_data_set(args, objects=[4,5,6,8]):
    '''
    Create the data set from the arguments.

    The Core 50 data set has:
    - 10 object classes
    - 5 object instances per class
    - 11 background conditions in which each of the object instances are imaged in
    

    The specific arrangement of operations defines the particular problem that we are solving.
    Specifically:
    - All 5 object instances occur in each fold
    - Folds are defined as pairs of background conditions (a total of 5 folds)

    So, we are learning a model that can distinguish between *these* specific object instances, but
    under arbitrary background conditions

    
    :param args: Command line arguments
    :param objects: List of objects to include in the training/evaluation process (integers 0...9)
    
    :return: TF Datasets for the training, validation and testing sets + number of classes
    '''

    # Test create object-based rotations
    core = Core50(args.meta_dataset)

    # Set the problem class IDs
    # Object 4->C0; Object 5->C1; Object 6->C2; Object 8->C3;
    #   ignore all other objects

    core.set_problem_class_by_equality('class', objects)

    # Select only these object classes (remove all others)
    core.filter_problem_class()

    # Folds by pairs of condition ((1,2), (3,4), ...)
    folds = core.create_subsets_by_membership('condition', list(zip(range(1,11,2),range(2,11,2))))

    # Check to make sure that argument matches that actual number of folds
    assert len(folds) == args.Nfolds, "args.Nfolds does not match actual number of folds"

    # Create training/validation/test DFs
    df_training, df_validation, df_testing = core.create_training_validation_testing(args.rotation,
                                                                                     folds,
                                                                                     args.Ntraining)

    print("Training set has %d samples"%(len(df_training.index)))
    print("Validation set has %d samples"%(len(df_validation.index)))
    if df_testing is None:
        print("Testing set has NO samples")
    else:
        print("Testing set has %d samples"%(len(df_testing.index)))
    
    # Create the corresponding Datasets
    ds_training = core.create_dataset(df_training, args.dataset,
                                      column_file='fname',
                                      column_label='problem_class',
                                      batch_size=args.batch,
                                      prefetch=args.prefetch,
                                      num_parallel_calls=args.num_parallel_calls,
                                      cache=args.cache,
                                      repeat=args.repeat,
                                      shuffle=args.shuffle,
                                      dataset_name='train')

    ds_validation = core.create_dataset(df_validation, args.dataset,
                                      column_file='fname',
                                      column_label='problem_class',
                                      batch_size=args.batch,
                                      prefetch=args.prefetch,
                                      num_parallel_calls=args.num_parallel_calls,
                                      cache=args.cache,
                                      repeat=False,
                                      shuffle=0,
                                      dataset_name='valid')

    if df_testing is None:
        ds_testing = None
    else:
        ds_testing = core.create_dataset(df_testing, args.dataset,
                                      column_file='fname',
                                      column_label='problem_class',
                                      batch_size=args.batch,
                                      prefetch=args.prefetch,
                                      num_parallel_calls=args.num_parallel_calls,
                                      cache=args.cache,
                                      repeat=False,
                                      shuffle=0,
                                      dataset_name='test')
            
    return ds_training, ds_validation, ds_testing, len(objects)



#################################################################
def load_data_set_by_folds(args, objects=[4,5,6,8], seed=42):
    '''
    Create the data set from the arguments.

    The underlying caching is done by fold (as opposed to data set)

    The Core 50 data set has:
    - 10 object classes
    - 5 object instances per class
    - 11 background conditions in which each of the object instances are imaged in
    

    The specific arrangement of operations defines the particular problem that we are solving.
    Specifically:
    - All 5 object instances occur in each fold
    - Folds are defined as pairs of background conditions (a total of 5 folds)

    So, we are learning a model that can distinguish between *these* specific object instances, but
    under arbitrary background conditions

    
    :param args: Command line arguments
    :param objects: List of objects to include in the training/evaluation process (integers 0...9)
    
    :return: TF Datasets for the training, validation and testing sets + number of classes
    '''

    # Test create object-based rotations
    core = Core50(args.meta_dataset)

    # Set the problem class IDs
    # Object 4->C0; Object 5->C1; Object 6->C2; Object 8->C3;
    #   ignore all other objects

    core.set_problem_class_by_equality('class', objects)

    # Select only these object classes (remove all others)
    core.filter_problem_class()

    # Folds by pairs of condition ((1,2), (3,4), ...)
    folds = core.create_subsets_by_membership('condition', list(zip(range(1,11,2),range(2,11,2))))

    # Shuffle the rows in each of the dataframes
    folds = [df.sample(frac=1, random_state=seed) for df in folds]

    # Check to make sure that argument matches that actual number of folds
    assert len(folds) == args.Nfolds, "args.Nfolds does not match actual number of folds"

    # Create the fold-wise data sets
    ds_folds = []
    for i, f in enumerate(folds):
        # Create the corresponding Datasets: only up to caching
        ds = core.create_dataset(f, args.dataset,
                                 column_file='fname',
                                 column_label='problem_class',
                                 batch_size=1,
                                 prefetch=0,
                                 num_parallel_calls=args.num_parallel_calls,
                                 cache=args.cache,
                                 repeat=False,
                                 shuffle=0,
                                 dataset_name='core50_objects_10_fold_%d'%i)
        ds_folds.append(ds)


    # Create training/validation/test DFs
    # This step does the batching/prefetching/repeating/shuffling
    ds_training, ds_validation, ds_testing = core.create_training_validation_testing_from_datasets(args.rotation,
                                                                                                   ds_folds,
                                                                                                   args.Ntraining,
                                                                                                   batch_size=args.batch,
                                                                                                   prefetch=args.prefetch,
                                                                                                   repeat=args.repeat,
                                                                                                   shuffle=args.shuffle)

    # Done
    return ds_training, ds_validation, ds_testing, len(objects)



    
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
    
    ####################################################
    # Build the model
    # image_size=args.image_size[0:2]
    # nchannels = args.image_size[2]

    # Network config
    # NOTE: this is very specific to our implementation of create_cnn_classifier_network()
    #   List comprehension and zip all in one place (ugly, but effective).
    #   Feel free to organize this differently
    dense_layers = [{'units': i} for i in args.hidden]
    
    conv_layers = [{'filters': f, 'kernel_size': (s,s), 'pool_size': (p,p), 'strides': (p,p)} if p > 1
                   else {'filters': f, 'kernel_size': (s,s), 'pool_size': None, 'strides': None}
                   for s, f, p, in zip(args.conv_size, args.conv_nfilters, args.pool)]
    
    print("Dense layers:", dense_layers)
    print("Conv layers:", conv_layers)


    if multi_gpus > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():

            # Build network: you must provide your own implementation
            model = create_srnn_classifier_network(data_size=data_out['len_max'],
                                          conv_layers=conv_layers,
                                          dense_layers=dense_layers,
                                          p_dropout=args.dropout,
                                          #p_spatial_dropout=args.spatial_dropout,
                                          lambda_l2=args.L2_regularization,
                                          lambda_l1=args.L1_regularization,
                                          lrate=args.lrate, n_classes=n_classes,
                                          loss='sparse_categorical_crossentropy',
                                          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                                          padding=args.padding,
                                          flatten=False,
                                          args=args)
    else:
            model = create_srnn_classifier_network(data_size=int(data_out['len_max']),
                                          conv_layers=conv_layers,
                                          dense_layers=dense_layers,
                                          p_dropout=args.dropout,
                                          # p_spatial_dropout=args.spatial_dropout,
                                          lambda_l2=args.L2_regularization,
                                          lambda_l1=args.L1_regularization,
                                          lrate=args.lrate, n_classes=n_classes,
                                          loss='sparse_categorical_crossentropy',
                                          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                                          padding=args.padding,
                                          flatten=False,
                                          args = args)

    
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
    
    return model


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
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
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
        
