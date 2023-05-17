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

# USE THE CPU NOT THE GPU - errored out when tried it in the main function and needws to be done right after reading TF 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
    
    
from hw6_base import *

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
    parser.add_argument('--path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw6/results',help = 'Provide path to the result file')
    # parser.add_argument('--base', type = str, default = 'bmi_*results.pkl', help= 'Provide Filename structure, may use * for wildcard')
    # parser.add_argument('--cnn_path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw5/results/shallow/run3/', help= 'Provide Path to CNN Network')
    # parser.add_argument('--srnn_path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw5/results/deep/run9/', help= 'Provide path to SRNN Network')
    # parser.add_argument('--rnn_pool_path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw5/results/deep/run9/', help= 'Provide path to RNN pool Network')
    parser.add_argument('--base', type = str, default = 'image*rot*_results.pkl', help= 'Provide Filename structure for results, may use * for wildcard')
    parser.add_argument('--gru_path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw6/results/deep/run9/', help= 'Provide path to GRU Network')
    parser.add_argument('--mha_path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw6/results/deep/run9/', help= 'Provide path to MHA Network')
    
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
    
    train_accuracy = []
    val_accuracy = []
    test_accuracy = []
    for i, result in enumerate(results):
        train_accuracy.append(result['history']['sparse_categorical_accuracy'])
        val_accuracy.append(result['history']['val_sparse_categorical_accuracy'])
        test_accuracy.append(result['predict_testing_eval'][1])
    
    return train_accuracy, val_accuracy, test_accuracy

def plot_results_new(data1, label1 = 'Hello' , data2 = None, label2 = None , data3 = None, label3 = None, data4 = None, label4 = None, graph_params = None):
    
    '''
    We are taking the data as a set of all the results for each model including all rotations and plotting them
    
    ''' 
    
    for data in data1:
        plt.plot(range(0,len(data)), data, label = label1 +'_Rot_' +str(data1.index(data)), alpha = 0.1 + 0.2*data1.index(data), color = 'red')
    if data2 is not None:
        for data in data2:
            plt.plot(range(0,len(data)), data, label = label2 +'_Rot_' +str(data2.index(data)), alpha = 0.1 + 0.2*data2.index(data), color = 'blue')
    if data3 is not None:
        for data in data3:
            plt.plot(range(0,len(data)), data, label = label3 +'_Rot_' +str(data3.index(data)), alpha = 0.1 + 0.2*data3.index(data), color = 'green')
    if data4 is not None:
        for data in data4:
            plt.plot(range(0,len(data)), data, label = label4 +'_Rot_' +str(data4.index(data)), alpha = 0.1 + 0.2*data4.index(data), color = 'yellow')
    
    plt.legend()
    if graph_params is not None:
        plt.title(graph_params['title'])
        plt.xlabel(graph_params['xlabel'])
        plt.ylabel(graph_params['ylabel'])

    plt.savefig("plots/Fig1_%s.png"%graph_params['title'])
    print('Figure Saved: %s'%graph_params['title'])
    plt.clf()
    
    return 0

def plot_scatter(data1, data2, graph_params):
    
    # Plot the scatter plot of the data
    plt.scatter(data1, data2, c=['red', 'blue', 'green', 'yellow', 'orange'], alpha = 0.5, s = [100, 100, 100, 100, 100])
    
    for i in range(len(data1)):
        plt.text(data1[i], data2[i], str(i), fontsize=12)  
    
    if graph_params is not None:
        plt.title(graph_params['title'])
        plt.xlabel(graph_params['xlabel'])
        plt.ylabel(graph_params['ylabel'])

    plt.savefig("plots/Fig3_%s.png"%graph_params['title'])
    print('Figure Saved: %s'%graph_params['title'])
    plt.clf()
    
    return 0

if __name__ == "__main__":
    
    # Hide GPU from visible devices
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    parser = create_parser()
    args = parser.parse_args()

    '''
    Handle the arguments to provide inputs to the function to build figures for HW 3
    
    1. We have path variables to provide the path to the results files
    2. We have deep_base and deep_path to provide the path to the deep learning results files
    3. We need to get the results and plot them 
    ''' 
    

    # Get the results for the all models
    gru = read_all_rotations(args.gru_path, args.base)
    mha = read_all_rotations(args.mha_path, args.base)
    
    # Make the results in plottable format
    
    gru_train_accuracy, gru_val_accuracy, gru_test_accuracy = prepare_result(gru)
    mha_train_accuracy, mha_val_accuracy, mha_test_accuracy = prepare_result(mha)
    # rnn_pool_train_accuracy, rnn_pool_val_accuracy, rnn_pool_test_accuracy = prepare_result(rnn_pool)
    
    # print(rnn_pool_test_accuracy)
    # make_test_predictions(deep,shallow) 
    gru_length = []
    mha_length = []
    for a in gru_train_accuracy:
        print(len(a))
        gru_length.append(len(a))

    for a in mha_train_accuracy:
        print(len(a))
        mha_length.append(len(a))

    '''
  Plot Figures 1 2 3 
  
  We are plotting a total of 3 figures here 
  
    1. Training set accuracy as a function of epoch for each rotation.
    2. Validation set accuracy as a function of epoch for each rotation.
    3. Scatter plot of the test set accuracy for each rotation.
  
  We had to make multiple iterations of plot_results because we have some amount of manual inputs going into the function here. 
  Figures and Plots need to be personalized for this task
    '''
    if args.plot:
    
        # Plot Figure 1
        graph_params = {'title': 'Training Set Accuracy', 'xlabel': 'Epochs', 'ylabel': 'Accuracy'}
        plot_results_new(gru_train_accuracy, 'GRU', mha_train_accuracy, 'MHA',  graph_params = graph_params)
        
        # Plot Figure 2
        graph_params = {'title': 'Validation Set Accuracy', 'xlabel': 'Epochs', 'ylabel': 'Accuracy'}
        plot_results_new(gru_val_accuracy, 'GRU', mha_val_accuracy, 'MHA',  graph_params = graph_params)
        
        # Figure 3
        
        #Compute Test Accuracy(CNN) - Test Accuracy(SRNN)
        
        graph_params = {'title': 'Test Set Accuracy for each rotation', 'xlabel': 'GRU', 'ylabel': 'MHA'}
        plot_scatter(gru_test_accuracy, mha_test_accuracy, graph_params = graph_params)

        graph_params = {'title': 'Number of Training Epochs', 'xlabel': 'GRU', 'ylabel': 'MHA'}
        plot_scatter(gru_length, mha_length, graph_params = graph_params)