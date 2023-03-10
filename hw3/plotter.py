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
    parser.add_argument('--path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw3/results',help = 'Provide path to the result file')
    parser.add_argument('--base', type = str, default = 'bmi_*results.pkl', help= 'Provide Filename structure, may use * for wildcard')
    parser.add_argument('--shallow_path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw3/results/shallow/run1/', help= 'Provide Path to dropout')
    parser.add_argument('--deep_path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw3/results/deep/run1/', help= 'Provide path to regularization')
    parser.add_argument('--shallow_base', type = str, default = 'image*rot*_results.pkl', help= 'Provide Filename structure for shallow results, may use * for wildcard')
    parser.add_argument('--deep_base', type = str, default = 'image*rot*_results.pkl', help= 'Provide Filename structure for Deep results, may use * for wildcard')
    # Print Figure 1 only
    parser.add_argument('--single_file', action='store_true', help='Perform on a single file')
    
    # types of outputs  
    parser.add_argument('--dropout', action='store_true', help='Plot results for dropout set')
    parser.add_argument('--regularization', action='store_true', help='Plot results for L1 regularization set')

    # 
    return parser
    
    
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
    
    return 0
    

def get_results(path, base):
    
    results = read_all_rotations(path, base)
    
    print(len(results))
    return results


if __name__ == "__main__":
    
    
    parser = create_parser()
    args = parser.parse_args()

    '''
    Handle the arguments to provide inputs to the function to build figures for HW 3
    
    1. We have path variables to provide the path to the results files
    2. We have deep_base and deep_path to provide the path to the deep learning results files
    3. We need to get the results and plot them 
    ''' 
    

    # Get the results for the deep models
    
    deep = get_results(args.deep_path, args.deep_base)
    shallow = get_results(args.shallow_path, args.shallow_base)