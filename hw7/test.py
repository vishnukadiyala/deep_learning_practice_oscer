import os
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import argparse
import sys
import time
import fnmatch
import scipy 
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

def read_all_rotations(dirname, filebase):
    '''Read results from dirname from files matching filebase'''

    # The set of files in the directory
    files = fnmatch.filter(os.listdir(dirname), filebase)
    test_predictions = []
    test_pred_eval = []
    # Loop over matching files
    for f in sorted(files):
        print(f)
        fp = open("%s/%s"%(dirname,f), "rb")
        r = pickle.load(fp)
        fp.close()
        print(r.keys())
        test_predictions.append(r['predict_testing'])
        test_pred_eval.append(r['predict_testing_eval'])
        
        # test_predictions.append(r['predict_testing'])

        # results.append(r)
    return test_predictions, test_pred_eval
def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Plotter Function', fromfile_prefix_chars='@')
    
    # Path instructions handler 
    parser.add_argument('--path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw6/results',help = 'Provide path to the result file')
    parser.add_argument('--base', type = str, default = 'image*rot*_results.pkl', help= 'Provide Filename structure for results, may use * for wildcard')
    parser.add_argument('--unet_path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw7/results/Unet/run9/', help= 'Provide path to UNET Network')
    parser.add_argument('--auto_path', type = str, default = '/home/vishnu/vscode/AML/deep_learning_practice_oscer/hw7/results/autoencoder/', help= 'Provide path to AUTOENCODER Network')
    parser.add_argument('--ds_path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw7/results/ds/run9/', help= 'Provide path to Testing Dataset')
    parser.add_argument('--ds_base', type = str, default = '*_test.pkl', help= 'Provide base to Testing Dataset')
    # Print Figures
    parser.add_argument('--plot', action='store_true', help='Plot results')

    # 
    return parser
if __name__ == "__main__":
    
    # Hide GPU from visible devices if you want to use CPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    parser = create_parser()
    args = parser.parse_args()
    tp, tpe = read_all_rotations(args.auto_path, args.base)
    
    print(tpe[0])

    #print(tp[0][0].shape)