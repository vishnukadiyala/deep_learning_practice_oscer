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
import scipy 

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
    parser.add_argument('--path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw2/results/r2/',help = 'Provide path to the result file')
    parser.add_argument('--base', type = str, default = 'bmi_*results.pkl', help= 'Provide Filename structure, may use * for wildcard')
    parser.add_argument('--dropout_path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw2/results/dropout3/', help= 'Provide Path to dropout')
    parser.add_argument('--reg_path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw2/results/reg2/', help= 'Provide path to regularization')
    
    # Print Figure 1 only
    parser.add_argument('--single_file', action='store_true', help='Perform on a single file')
    
    # types of outputs  
    parser.add_argument('--dropout', action='store_true', help='Plot results for dropout set')
    parser.add_argument('--regularization', action='store_true', help='Plot results for L1 regularization set')

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
    plt.savefig("plots/Fig1.png")

def get_train_folds(dir, base):
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
    # to fix that we need to add the first 2 elements at the end 
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

    return (train_fvaf, val_fvaf, test_fvaf, folds)
    # # plot the figure 
    
    # plt.plot(folds, train_fvaf, label = 'Training FVAF')
    # plt.plot(folds, val_fvaf, label = 'Validation FVAF')
    # #plt.plot(folds, test_fvaf, label = 'Testing FVAF')
    # plt.legend()
    # plt.ylabel('Avg. FVAF')
    # plt.xlabel('Folds')
    # plt.savefig("plots/Fig1_%s.png"%base)
    
def plot_results(folds = None, data1 = None, data2 = None, data3 = None, data4 = None, xlabel = None, ylabel= None, title = None):
    """
    This function builds plots based on the inputs given

    """
    if data1 is not None:
        (data, label1) = data1
        #print(data)
        plt.plot(folds, data, label = label1 )

    if data2 is not None:
        (data, label2) = data2
        plt.plot(folds, data, label = label2)

    if data3 is not None:
        (data, label3) = data3
        plt.plot(folds, data, label = label3)

    if data4 is not None:
        (data, label4) = data4
        plt.plot(folds, data, label = label4)

    plt.legend()
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if title is not None:
        #print(title)
        plt.title(title)

    plt.savefig("plots/Fig4_%s.png"%title)
    print("Figure Saved!")
    plt.clf()
    

def read_one_Ntrainig(dirname, filebase):
    
    test_fvaf =[]
    
    results = read_all_rotations(dirname, filebase)
    
    for result in results:
        test_fvaf.append(result['predict_testing_eval'][1])
    
    return test_fvaf
    
    


if __name__ == "__main__":
    
    
    parser = create_parser()
    args = parser.parse_args()

    drop_bases = ["bmi_*_dropout_0.1_results.pkl", "bmi_*_dropout_0.25_results.pkl", "bmi_*_dropout_0.5_results.pkl", "bmi_*_dropout_0.75_results.pkl"]
    reg_bases = ["bmi_*_regularization_0.1_results.pkl", "bmi_*_regularization_0.01_results.pkl", "bmi_*_regularization_0.001_results.pkl", "bmi_*_regularization_0.0001_results.pkl"]
    
    dropout_results = []
    reg_results = []
    norm_results = []
    max_d =[]
    max_l =[]


    if(args.single_file):
        display_prediction(args)
        exit()

    # get results for all the normal runs
    train_fvaf, val_fvaf, test_fvaf, folds = get_train_folds(args.path, args.base)
    fvaf = {'train':train_fvaf, 'val':val_fvaf, 'test':test_fvaf}
    norm_results.append(fvaf)
    
    # plot curves for no regularization
    plot_results(folds, data1 = (norm_results[0]['train'], 'Training'),data2 = (norm_results[0]['val'], 'Validation'),xlabel = 'Folds', ylabel = 'FVAF', title = 'No Regularization')
    
    # get results for all the dropout runs
    if (args.dropout):
        for i,n in enumerate(drop_bases):
            train_fvaf, val_fvaf, test_fvaf, folds = get_train_folds(args.dropout_path, n)
            fvaf = {'train':train_fvaf, 'val':val_fvaf, 'test':test_fvaf}
            dropout_results.append(fvaf)
            max_d.append(np.max(fvaf['val']))
        #best_dropout = np.argmax(max_d)
        #print("Best Dropout : ", best_dropout)
        #print(drop_bases[best_dropout])
        
        # plot results for dropout 
        plot_results(folds, data1 = (dropout_results[0]['val'], 'Dropout 0.1'),data2 = (dropout_results[1]['val'], 'Dropout 0.25'),data3 = (dropout_results[2]['val'], 'Dropout 0.5'), data4 = (dropout_results[3]['val'], 'Dropout 0.75'), xlabel = 'Folds', ylabel = 'Mean FVAF', title = 'Dropout')
    
    #get results for all the regularizarion runs
    if (args.regularization):
        for i,n in enumerate(reg_bases):
            train_fvaf, val_fvaf, test_fvaf, folds = get_train_folds(args.reg_path, n)
            fvaf = {'train':train_fvaf, 'val':val_fvaf, 'test':test_fvaf}
            max_l.append(np.max(fvaf['val']))
            reg_results.append(fvaf)
        #best_lx = np.argmax(max_l)
        #print("Best L1 : ",best_lx)
        #print(reg_bases[best_lx])
        
        # plot results for all regularization runs 
        plot_results(folds, data1 = (reg_results[0]['val'], 'L1 0.1'),data2 = (reg_results[1]['val'], 'L1 0.01'),data3 = (reg_results[2]['val'], 'L1 0.001'), data4 = (reg_results[3]['val'], 'L1 0.0001'), xlabel = 'Folds', ylabel = 'Mean FVAF', title = 'L1 regularization')

    # Build results and prepare for plotting 
    if (args.dropout and args.regularization):
        # Perform t-test 
        temp = []
        for item in dropout_results:
            temp.append(item['val'])
            temp1 = np.array(temp)
            best_dropout = np.argmax(temp1, axis = 0)
        print(best_dropout)
        
        temp = []
        for item1 in reg_results:
            temp.append(item1['val'])
            temp1 = np.array(temp)
            best_lx = np.argmax(temp1, axis = 0)
        print(best_lx)
        
    # Build results for plotting Fig 4

    test_dropout = []
    test_reg = []
    for i, n in enumerate(best_dropout):
        test_dropout.append(dropout_results[n]['test'][i])
    
    for i, n in enumerate(best_lx):
        test_reg.append(reg_results[n]['test'][i])
    
    test_normal = norm_results[0]['test']
    
    plot_results(folds, data1 = (test_dropout, "Dropout"), data2 = (test_normal, "No regularization"), data3 = (test_reg, "L1 Regularization"), xlabel = "Folds", ylabel = "Best fvaf", title = "Best hyperparameters" )
    
