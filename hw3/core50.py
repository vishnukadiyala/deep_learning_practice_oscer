'''
Tools for translating a description of the core50 data set into TF Datasets for
training and evaluating models.

core50_df is a DataFrame that contains one row per sample in the entire data set.
- condition: int describing the condition number (1..12)
- object: int describing unique object id
- fname: str of the file name relative to the main core 50 directory
- class: object class number (0 ... 9) in the full data set
- example: example # of the object instance within its class (0...4)
- problem_class: str assigned class ID for the purposes of learning (Cddd; ddd= class #)

The provided tools allow one to:
- Set the class number as a function of other properties in the table
- Remove all rows that do not have an assigned class number
- Create a set of folds given properties in the table (a list of DataFrames)
- Create a TF Dataset from a DataFrame

Andrew H. Fagg
andrewhfagg@gmail.com
'''

import png
import os
import fnmatch
import re
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

class Core50:
    def __init__(self, core50_fname:str='core50_df.pkl'):
        '''
        Constructor

        :param core50_fname: Name of the pickle file that contains the full dataset DataFrame
        '''
        self.df = None

        # Load the DF that describes the entire dataset
        with open(core50_fname, 'rb') as fp:
            self.df = pickle.load(fp)

    def create_subsets_by_membership(self, column:str, subsets:[[int]],
                                     from_df:pd.DataFrame=None)->[pd.DataFrame]:
        '''
        Create a set of DataFrames based on membership within a set of integers

        :param column: Column in the main DataFrame to check membership against
        :param subsets: List of a list of ints.  Specifies which rows in the main DataFrame to
               include in each of the output DataFrames.  If the column value is contained in the
               first list of ints, then that row is included in the first DataFrame, etc.
        :param from_df: DataFrame to use as the main DataFrame (if None, use the internally stored one)
        :return: List of DataFrames
        '''

        # List to return
        out = []

        # Resolve which DF to use
        if from_df is None:
            from_df = self.df

        # Iterate over the subsets
        for s in subsets:
            # For this subset, select rows that whose column values is in this subset
            out.append(self.df[self.df[column].isin(s)])

        return out

    def create_subsets_by_equality(self, column:str, subsets:[int], from_df:pd.DataFrame=None)->[pd.DataFrame]:
        '''
        Create a set of DataFrames based equality of the ints in subsets

        :param column: Column in the main DataFrame to check membership against
        :param subsets: List of ints.  Specifies which rows in the main DataFrame to
               include in each of the output DataFrames.  If the column value is equal to 
               the first int, then that row is included in the first DataFrame, etc.
        :param from_df: DataFrame to use as the main DataFrame (if None, use the internally stored one)
        :return: List of DataFrames
        '''
        # List to return
        out = []

        # Resolve which main DF to use
        if from_df is None:
            from_df = self.df

        # Iterate over the subsets
        for s in subsets:
            # Create a new DataFrame whose column value matches this subset int
            out.append(from_df[from_df[column] == s])
            
        return out

    def set_problem_class_by_membership(self, column:str, subsets:[[int]],
                                        from_df:pd.DataFrame=None, offset:int=0):
        '''
        Set the problem_class column for a set of rows in the specified DF based on membership
        in a set of ints.

        :param column: Column in the main DataFrame to check membership against
        :param subsets: List of a list of ints.  Specifies which rows in the main DataFrame to
               change.  If the column value is contained in the
               first list of ints, then the class is set to C0 (+offset), etc.
        :param from_df: DataFrame to use as the main DataFrame (if None, use the internally stored one)
        :param offset: Class index offset
        
        '''
        # Resolve which is the main DF
        if from_df is None:
            from_df = self.df

        # Iterate over the subsets
        for i, s in enumerate(subsets):
            # For this subset, if the column is in the subset, then set
            #  its problem_class column to "Cddd", where ddd is the subset # (+offset)
            from_df.loc[from_df[column].isin(s), "problem_class"] = 'C%d'%(i+offset)


    def set_problem_class_by_equality(self, column:str, subsets:[int],
                                      from_df:pd.DataFrame=None, offset:int=0):
        '''
        Set the problem_class column for a set of rows in the specified DF based on equalty to an int.

        :param column: Column in the main DataFrame to check membership against
        :param subsets: List of a ints.  Specifies which rows in the main DataFrame to
               change.  If the column value is equal to the first int, then the class
               is set to C0 (+offset), etc.
        :param from_df: DataFrame to use as the main DataFrame (if None, use the internally stored one)
        :param offset: Class index offset
        
        '''
        # Resolve which is the main DF
        if from_df is None:
            from_df = self.df

        # Iterate over the subsets
        for i, s in enumerate(subsets):
            # If the column matches the subset int, then set class to Cddd, where
            #  ddd is the integer position in the subsets array (+offset)
            from_df.loc[from_df[column] == s, "problem_class"] = 'C%d'%(i+offset)

    def filter_problem_class(self, from_df:pd.DataFrame=None):
        '''
        Remove all rows in a dataframe that do not have a problem_class defined.
        :param from_df: DF to operate on (None -> use the internal DF)
        :return: The new DF (note that the internal DF is also destructively modified)
        '''
        
        if from_df is None:
            # Internal DF
            self.df = self.df[self.df["problem_class"] != '']
            return self.df
        else:
            # Some other DF
            from_df = from_df[from_df["problem_class"] != '']
            return from_df
            

    def create_training_validation_testing(self, rotation:int, folds:[pd.DataFrame],
                                           ntraining:int, seed:int=42)->[pd.DataFrame]:

        '''
        Translate a list of DataFrames describing folds into DataFrames that represent the training,
        validation and testing sets, respectively

        :param rotation: Integer rotation (0 <= rot < nobjects)
        :param folds: array of folds described as DataFrames
        :param ntraining: Number of training folds
           If ntraining <= # folds-2, then we will have both validation and test sets
           If ntraining == # folds-1, then we will have only a validation set
        :param seed: Random seed for shuffling the data
        :return: DataFrames for each of the training, validation and testing sets
        '''
        
        # Number of folds
        nfolds = len(folds)

        assert (ntraining < nfolds), "Ntraining must be less than number of folds"
        assert (ntraining > 0), "Ntraining must be at least 1"

        if ntraining == nfolds - 1:
            # Fold indices for training and validation sets
            folds_training = (np.arange(ntraining) + rotation) % nfolds
            folds_validation = (nfolds - 1 + rotation ) % nfolds
            folds_testing = None
            df_testing = None
        else:
            # Fold indices for training, validation and test sets
            folds_training = (np.arange(ntraining) + rotation) % nfolds
            folds_validation = (nfolds - 2 + rotation ) % nfolds
            folds_testing = (nfolds - 1 + rotation ) % nfolds
            df_testing = pd.DataFrame()

        print("Training: ", folds_training)
        print("Validation: ", folds_validation)
        print("Testing: ", folds_testing)
    
        # Initially empty dataframes for all data sets
        df_training = pd.DataFrame()
        df_validation = pd.DataFrame()

        # Iterate over folds in training set
        for i in folds_training:
            # Append new object to training set
            df_training = pd.concat([df_training, folds[i]])

        df_validation = folds[folds_validation]

        # If there is also a test set, then set it up
        if folds_testing is not None:
            df_testing = folds[folds_testing]

        # Shuffle the data
        df_training = df_training.sample(frac=1, random_state=seed)
        df_validation = df_validation.sample(frac=1, random_state=seed)
    
        if df_testing is not None:
            df_testing = df_testing.sample(frac=1, random_state=seed)
    
        # Done
        return df_training, df_validation, df_testing


    @staticmethod
    def load_single_png_image(base_dir: str, fname: str) -> tf.Tensor:
        '''
        Load a single image from disk
    
        :param base_dir: Base directory
        :param fname: Path relative to the base directory
        :return: TF Tensor (rxcx3)
    
        '''
        # Append the base_dir if it is a string
        if base_dir is not None:
            fname = base_dir + '/' + fname

        # We are using tf operators exclusively so we can map onto a GPU
    
        # Read the raw data
        image_string = tf.io.read_file(fname)

        # Interpret it as a PNG file
        image = tf.image.decode_png(image_string, channels=3)

        # Convert to standard TF Tensor format
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.squeeze(image)

        return image

    @staticmethod
    def prepare_single_example(base_dir: str, example: tf.Tensor) -> [tf.Tensor]:
        '''
        Given a single example from the core50 dataset, produce a tensorflow-ready example
            
        :param base_dir: Base directory where the examples will be loaded from
        :param example: A TF vector that contains two strings: path to a specific file and the class name.
           The class name is of the form "Cdddd", where dddd is an integer that must be parsed

        :return: A proper TF-ready example: array of two Tensors representing an input/output pair
        '''
            
        # Filename
        fname = example[0]

        # Parse class
        cl = example[1] 
        cl = tf.strings.to_number(tf.strings.substr(cl, 1, 10, unit='UTF8_CHAR'))
        cl = tf.cast(cl, tf.int8)

        # Load image
        img = Core50.load_single_png_image(base_dir, fname)
    
        return img, cl

    @tf.autograph.experimental.do_not_convert
    def create_dataset(self,
                       df: pd.DataFrame, base_dir: str,
                       column_file: str = 'file', column_label: str = 'label',
                       batch_size: int=8, prefetch: int=2, num_parallel_calls: int=4,
                       cache: str=None,
                       repeat: bool =False, shuffle: int =0)->tf.data.Dataset:
        '''
        Translate a dataframe containing image name/class pairs into a TF Dataset for training/evaluation

        :param df: Dataframe containing local or global file names and class labels (form = "Cdddd", where
        dddd is an integer number corresponding to the class)
        :param base_dir: String defining the base directory that the file names in df are referenced from
        :param column_file: Name of the df column that contains the file name
        :param column_label: Name of the df column that contains the label (str: Cddd, where ddd is the class number)
        :param batch_size: Size of the batches that this dataset produces
        :param prefetch: Size of the prefetch buffer
        :param num_parallel_calls: Number of threads to use to fill in the buffer
        :param cache: None -> no cache, "" -> cache to memory, other string: name of the file to cache to
        :param repeat: The output dataset should repeat indefinitely
        :param shuffle: 0 -> no shuffle, other positive integer -> shuffle buffer size

        :return: Dataset
    
        '''

        # Convert DF to a Dataset that contains the DF file name, class pairs
        ds = tf.data.Dataset.from_tensor_slices(df[[column_file, column_label]].to_numpy())

        # Convert to a Dataset that contains the image, class number examples
        ds = ds.map(lambda x: tf.py_function(func=Core50.prepare_single_example, inp=[base_dir, x],
                                             Tout=(tf.float32, tf.int8)),
                    num_parallel_calls=num_parallel_calls)

        # Create a cache?
        if cache is not None:
            if cache == "":
                # Cache to RAM
                print('Cache to RAM')
                ds = ds.cache()
            else:
                # Cache to a file
                print('Cache to file (%s)'%(cache))
                ds = ds.cache(cache)

        # Repeat the dataset indefinitely?
        if repeat:
            ds = ds.repeat()

        # Shuffle the samples?
        if shuffle > 0:
            ds = ds.shuffle(shuffle)
        
        # Batching
        ds = ds.batch(batch_size)

        # Buffer multiple batches
        ds = ds.prefetch(prefetch)
    

        return ds

    @staticmethod
    def create_example_dataset_1(rotation:int=0, dir_base:str='/home/fagg/datasets/core50/core50_128x128'):
        '''
        Create data set with 3 objects in the training set.

        Folds are object-based (one object per fold)

        Use nfolds-2 folds as training folds

        :param rotation: Rotation number to produce
        :param dir_base: Location of the core50 data set
        :return: TF Datasets: (training, validation, testing)
        '''
        # Test create object-based rotations
        core = Core50()

        # Set the problem class IDs
        # Object 4->C0; Object 5->C1; Object 8->C2; ignore all other objects
        core.set_problem_class_by_equality('class', [4,5,8])

        # Select only these object classes
        core.filter_problem_class()

        # Folds by example within class
        #  Example is the object # within the class
        folds = core.create_subsets_by_equality('example', list(range(5)))

        # Create training/validation/test DFs
        df_training, df_validation, df_testing = core.create_training_validation_testing(rotation,
                                                                                         folds,
                                                                                         len(folds)-2)

        # Create Datasets
        ds_training = core.create_dataset(df_training, dir_base, column_file='fname',
                                          column_label='problem_class', shuffle=100)

        ds_validation = core.create_dataset(df_validation, dir_base, column_file='fname',
                                            column_label='problem_class', shuffle=100)

        if df_testing is None:
            ds_testing = None
        else:
            ds_testing = core.create_dataset(df_testing, dir_base, column_file='fname',
                                             column_label='problem_class', shuffle=100)
            
        return ds_training, ds_validation, ds_testing

    @staticmethod
    def create_example_dataset_2(rotation:int=0, dir_base:str='/home/fagg/datasets/core50/core50_128x128'):
        '''
        Create data set with all 5 objects in the training set.

        Folds are condition based (one condition per fold).  Using 10 conditions (so, 10 folds)

        Use nfolds-2 folds as training folds

        :param rotation: Rotation number to produce
        :param dir_base: Location of the core50 data set
        :return: TF Datasets: (training, validation, testing)
        '''
        
        # Test create object-based rotations
        core = Core50()

        # Set the problem class IDs
        # Object 4->C0; Object 5->C1; Object 8->C2; ignore all other objects
        core.set_problem_class_by_equality('class', [4,5,8])

        # Select only these object classes
        core.filter_problem_class()

        # Folds by condition
        folds = core.create_subsets_by_equality('condition', list(range(1,11)))

        # Create training/validation/test DFs
        df_training, df_validation, df_testing = core.create_training_validation_testing(rotation,
                                                                                         folds,
                                                                                         len(folds)-2)

        # Create Datasets
        ds_training = core.create_dataset(df_training, dir_base, column_file='fname',
                                          column_label='problem_class', shuffle=100)

        ds_validation = core.create_dataset(df_validation, dir_base, column_file='fname',
                                            column_label='problem_class', shuffle=100)

        if df_testing is None:
            ds_testing = None
        else:
            ds_testing = core.create_dataset(df_testing, dir_base, column_file='fname',
                                             column_label='problem_class', shuffle=100)
            
        return ds_training, ds_validation, ds_testing

    @staticmethod
    def create_example_dataset_3(rotation:int=0, dir_base:str='/home/fagg/datasets/core50/core50_128x128'):
        '''
        Create data set with all 5 objects in the training set.

        Folds are condition based (two conditions per fold).  Using 10 conditions (so, 5 folds)

        Use nfolds-2 folds as training folds

        :param rotation: Rotation number to produce
        :param dir_base: Location of the core50 data set
        :return: TF Datasets: (training, validation, testing)
        '''
        
        # Test create object-based rotations
        core = Core50()

        # Set the problem class IDs
        # Object 4->C0; Object 5->C1; Object 8->C2; ignore all other objects
        core.set_problem_class_by_equality('class', [4,5,8])

        # Select only these object classes
        core.filter_problem_class()

        # Folds by pairs of condition
        folds = core.create_subsets_by_membership('condition', list(zip(range(1,11,2),range(2,11,2))))

        print(len(folds))
        
        # Create training/validation/test DFs
        df_training, df_validation, df_testing = core.create_training_validation_testing(rotation,
                                                                                         folds,
                                                                                         len(folds)-2)

        # Create Datasets
        ds_training = core.create_dataset(df_training, dir_base, column_file='fname',
                                          column_label='problem_class', shuffle=100)

        ds_validation = core.create_dataset(df_validation, dir_base, column_file='fname',
                                            column_label='problem_class', shuffle=100)

        if df_testing is None:
            ds_testing = None
        else:
            ds_testing = core.create_dataset(df_testing, dir_base, column_file='fname',
                                             column_label='problem_class', shuffle=100)
            
        return ds_training, ds_validation, ds_testing


