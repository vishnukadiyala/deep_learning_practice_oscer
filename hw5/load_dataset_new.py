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

