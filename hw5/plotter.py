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
import tensorflow as tf
from core50 import *
from hw3_base import *

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
    parser.add_argument('--shallow_path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw3/results/shallow/run3/', help= 'Provide Path to Shallow Network')
    parser.add_argument('--deep_path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw3/results/deep/run9/', help= 'Provide path to Deep Network')
    parser.add_argument('--shallow_base', type = str, default = 'image*rot*_results.pkl', help= 'Provide Filename structure for shallow results, may use * for wildcard')
    parser.add_argument('--deep_base', type = str, default = 'image*rot*_results.pkl', help= 'Provide Filename structure for Deep results, may use * for wildcard')
    # Print Figures
    parser.add_argument('--plot', action='store_true', help='Plot results')
    
    # types of outputs  
    #parser.add_argument('--dropout', action='store_true', help='Plot results for dropout set')
    #parser.add_argument('--regularization', action='store_true', help='Plot results for L1 regularization set')

    # 
    return parser
    
    
def plot_results(epochs = None, data1 = None, data2 = None, data3 = None, data4 = None, data5 = None, xlabel = None, ylabel= None, title = None):
    """
    This function builds plots based on the inputs given

    """
    if data1 is not None:
        (data, label1) = data1
        #print(data)
        plt.plot(range(0,len(data)), data, label = label1 )

    if data2 is not None:
        (data, label2) = data2
        plt.plot(range(0,len(data)), data, label = label2)

    if data3 is not None:
        (data, label3) = data3
        plt.plot(range(0,len(data)), data, label = label3)

    if data4 is not None:
        (data, label4) = data4
        plt.plot(range(0,len(data)), data, label = label4)
    
    if data5 is not None:
        (data, label5) = data5
        plt.plot(range(0,len(data)), data, label = label5)

    plt.legend()
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if title is not None:
        #print(title)
        plt.title(title)

    plt.savefig("plots/Fig1_%s.png"%title)
    print('Figure Saved: %s'%title)
    plt.clf()
    
    return 0

def plot_hist(
    data1 = None, 
    data2 = None, 
    title = "Hello"
):
    if data2 is not None:
        x = np.array([data1,data2])
        x = x.T
    else:
        x = data1
        
    colors = ['red', 'blue']
    plt.hist(x, 7, density=False, histtype='step', color=colors, label = ("shallow", "deep"), fill = True, stacked = False, alpha = 0.5)
    plt.title("Accuracy between Shallow and Deep Networks")
    plt.legend()
    # plt.set_title('stacked bar')
    plt.savefig("plots/Fig3_%s.png"%title)
    plt.clf()
    print('Figure Saved: %s'%title)
    return 0

def make_test_predictions(results_deep, results_shallow):
    
    '''
    This Function takes the results file as an input and makes predictions on the test data, 
    It will also create a matplotlib figure and save it to the plots folder
    The plot will consist of 3 subplots along with the histogram of the test predictions. 
    '''
    
    a = 0
    # Get the best Validation Accuracy model
    for i, result in enumerate(results_deep):
        b = np.max(result['history']['val_sparse_categorical_accuracy'])
        if b > a:
            a = b
            best_model = result
    
    for i, result in enumerate(results_shallow):
        b = np.max(result['history']['val_sparse_categorical_accuracy'])
        if b > a:
            a = b
            best_model2 = result
    # print(best_model['fname_base'])
    # print(a)
    # Realized we dont need to load the model, we can just use the model that was saved in the results file
    model = tf.keras.models.load_model(best_model['fname_base'] + '_model')
    model2 = tf.keras.models.load_model(best_model2['fname_base'] + '_model')
    
    deep = []
    shallow = []
    # Load the testing data saved before
    data = tf.data.Dataset.load("/home/cs504305/deep_learning_practice/homework/hw3/test_ds")
    image = data.take(1)
    images, labels = tuple(zip(*data))
    images = np.array(images)
    labels = np.array(labels)
    plt.rcParams['figure.figsize'] = (10, 20)
    plt.rcParams['font.size'] = 8
    figure, axis = plt.subplots(4, 4)
    
    for i in range(4):
        for j in range(4):
        
            if i == 3 and j == 3:
                x = np.array([deep,shallow])
                x = x
                colors = ['red']
                axis[i,j].hist(x, density=False, histtype='step', color=colors, label = ("shallow", "deep"), fill = True, stacked = False, alpha = 0.5)
                # axis[i,1].set_title("Histogram of Predictions")
            else:
                num = np.random.randint(0, 10)
                x = model.evaluate(images[i])
                y = model2.evaluate(images[i])
                
                deep.append(x[0])
                shallow.append(y[0])
                axis[i, j].imshow(images[i][num])
                axis[i, j].set_title("Image_%s"%i)    
    
    plt.savefig("plots/Fig4.png")
    return 0

def prepare_result(results):
    
    '''
    
    This Function takes the result file as an input and prepares the data for plotting
    
    '''
    
    # Create data for plotting 
    val_loss = []
    val_accuracy = []
    test_accuracy = []
    for i, result in enumerate(results):
        val_loss.append(result['history']['val_loss'])
        val_accuracy.append(result['history']['val_sparse_categorical_accuracy'])
        test_accuracy.append(result['predict_testing_eval'][1])
    
    return val_loss, val_accuracy, test_accuracy

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
    deep = read_all_rotations(args.deep_path, args.deep_base)
    shallow = read_all_rotations(args.shallow_path, args.shallow_base)
    
    # Make the results in plottable format
    deep_val_loss, deep_val_accuracy, deep_test_accuracy = prepare_result(deep)
    shallow_val_loss, shallow_val_accuracy, shallow_test_accuracy = prepare_result(shallow)
    
    make_test_predictions(deep,shallow) 

    '''
  Plot Figures 1 2 3 4 
  
  We are plotting a total of 6 figures here 
  
  1. Validation Loss of Deep Networks, Across all rotations 
  2. Validation Accuracy of Deep Networks, Across all rotations
  3. Validation Loss of Shallow Networks, Across all rotations 
  4. Validation Accuracy of Shallow Networks, Across all rotations
  5. Test Accuracy of Deep Networks and shallow networks, Across all rotations
  6. predicted test labels of Deep Networks and shallow networks, Across few images
  
  We had to make multiple iterations of plot_results because we have some amount of manual inputs going into the function here. 
  Figures and Plots need to be personalized for this task
    '''
    if args.plot:
        plot_results(
            epochs = [range(1000)],
            data1 = (deep_val_loss[0], "Deep_rot_00"),
            data2 = (deep_val_loss[1], "Deep_rot_01"),
            data3 = (deep_val_loss[2], "Deep_rot_02"),
            data4 = (deep_val_loss[3], "Deep_rot_03"),
            data5 = (deep_val_loss[4], "Deep_rot_04"),
            xlabel = "Epochs",
            ylabel = "Val_loss",
            title = "Deep Network Validation Loss across Rotations"
            
        )

        
        plot_results(
            epochs = [range(1000)],
            data1 = (deep_val_accuracy[0], "Deep_rot_00"),
            data2 = (deep_val_accuracy[1], "Deep_rot_01"),
            data3 = (deep_val_accuracy[2], "Deep_rot_02"),
            data4 = (deep_val_accuracy[3], "Deep_rot_03"),
            data5 = (deep_val_accuracy[4], "Deep_rot_04"),
            xlabel = "Epochs",
            ylabel = "Val_Accuracy",
            title = "Deep Network Validation Accuracy across Rotations"
            
        )
        
        
        plot_results(
            epochs = [range(1000)],
            data1 = (shallow_val_loss[0], "Shallow_rot_00"),
            data2 = (shallow_val_loss[1], "Shallow_rot_01"),
            data3 = (shallow_val_loss[2], "Shallow_rot_02"),
            data4 = (shallow_val_loss[3], "Shallow_rot_03"),
            data5 = (shallow_val_loss[4], "Shallow_rot_04"),
            xlabel = "Epochs",
            ylabel = "Val_Loss",
            title = "Shallow Network Validation Loss across Rotations"
            
        )
        
        plot_results(
            epochs = [range(1000)],
            data1 = (shallow_val_accuracy[0], "Shallow_rot_00"),
            data2 = (shallow_val_accuracy[1], "Shallow_rot_01"),
            data3 = (shallow_val_accuracy[2], "Shallow_rot_02"),
            data4 = (shallow_val_accuracy[3], "Shallow_rot_03"),
            data5 = (shallow_val_accuracy[4], "Shallow_rot_04"),
            xlabel = "Epochs",
            ylabel = "Val_Accuracy",
            title = "Shallow Network Validation Accuracy across Rotations"
            
        )

        # Plot Figure 3 
        
        plot_hist(
            data1 = shallow_test_accuracy,
            data2 = deep_test_accuracy,
            title = "Test Accuracy comparision"
        )
        
        # Plot Figure 4 
        
        #plot_images()