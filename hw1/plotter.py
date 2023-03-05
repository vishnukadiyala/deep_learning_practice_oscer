'''
Plotter helps us plot beautiful figures from the results file

'''

import os
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import argparse
import sys
import time
import fnmatch

from hw1_base import check_args

#################################################################
# Default plotting parameters
FONTSIZE = 14
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = FONTSIZE

#################################################################

def read_all_rotations(dirname, filebase):
    '''Read results from dirname from files matching filebase'''

    # The set of files in the directory
    files = fnmatch.filter(os.listdir(dirname), filebase)
    files.sort()
    results = []

    # Loop over matching files
    for f in files:
        fp = open("%s/%s"%(dirname,f), "rb")
        r = pickle.load(fp)
        fp.close()
        results.append(r)
    return results

def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Plotter Function', fromfile_prefix_chars='@')
    
    # Path instructions handler 
    parser.add_argument('--path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw1/results/r1/',help = 'Provide path to the result file')
    parser.add_argument('--base', type = str, default = 'bmi_ddtheta_0_hidden_1000_JI_Ntraining_2_rotation_0_results.pkl', help= 'Provide Filename structure, may use * for wildcard')
    
    # Print Figure 1 only
    parser.add_argument('--single_file', action='store_true', help='Perform on a single file')
    
    
    # 
    return parser
    
def display_prediction(fbase):
    '''
    Display the predictions vs Actual values in fname
    '''
    
    # Load the history file and display it
    # argstring = args2string(args)
    fp = open(args.path + args.base, "rb")
    history = pickle.load(fp)
    fp.close()
    #print(history['outs_testing'])
    
    # Display
    plt.plot(history['time_testing'][0:500], history['outs_testing'][0:500], label = 'Actual acc.')
    plt.plot(history['time_testing'][0:500], history['predict_testing'][0:500], label = 'Predicted acc.')
    plt.legend()
    plt.ylabel('Shoulder acc.')
    plt.xlabel('time')
    plt.savefig("Fig1.png")

def display_train_folds(dir, base):
    '''
    Plot the training vs avg fvaf values for a set of results
    
    :param base: Directory containing a set of results files
    '''
    # Temporary list to perform sorting
    temp_train = []
    temp_val = []
    temp_test = []
    
    # Lists for plotting the data 
    train_fvaf = []
    val_fvaf = []
    test_fvaf = []
    folds = []
    # initialize 
    old_fold = 0
    
    # Read all the results
    results = read_all_rotations(dir, base)


    # Iterate through the results
    for result in results:
        history = result['history']
        new_fold = (len(result['folds']['folds_training']))
        
        if(new_fold!= old_fold):
            folds.append(new_fold)
            if (old_fold != 0):
                #print(new_fold)
                train_fvaf.append(np.mean(temp_train))
                val_fvaf.append(np.mean(temp_val))
                test_fvaf.append(np.mean(temp_test))
                
                
                temp_train = []
                temp_test = []
                temp_val = []
                
            old_fold = new_fold
        
        temp_train.append(result['predict_training_eval'][1])
        temp_val.append(result['predict_validation_eval'][1])
        temp_test.append(result['predict_testing_eval'][1])
    
    train_fvaf.append(np.mean(temp_train))
    val_fvaf.append(np.mean(temp_val))
    test_fvaf.append(np.mean(temp_test))                

    # Since we got the first 2 folds as 13 and 18 we are getting a weird graph
    # to do that we need to add the first 2 elements at the end 
    for i in range(2):
        # Pop the first iteam
        x = train_fvaf.pop(0)
        y = val_fvaf.pop(0)
        z = test_fvaf.pop(0)
        f = folds.pop(0)
        
        # add it to the end
        train_fvaf.append(x)
        val_fvaf.append(y)
        test_fvaf.append(z)
        folds.append(f)
        
    print(folds)
    # plot the figure 
    
    plt.plot(folds, train_fvaf, label = 'Training FVAF')
    plt.plot(folds, val_fvaf, label = 'Validation FVAF')
    plt.plot(folds, test_fvaf, label = 'Testing FVAF')
    plt.legend()
    plt.ylabel('Avg. FVAF')
    plt.xlabel('Folds')
    plt.savefig("Fig2.png")
    

if __name__ == "__main__":
    
    parser = create_parser()
    args = parser.parse_args()
    
    if(args.single_file):
        display_prediction(args)

    else:
        args.base = "bmi_ddtheta_0_hidden*_results.pkl"
        display_train_folds(args.path, args.base)