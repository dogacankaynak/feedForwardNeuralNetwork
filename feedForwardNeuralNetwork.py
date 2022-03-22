# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:17:58 2022

@author: dogacan.kaynak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import sys
"""
Activation function for FFNN is linear sigmoid
"""
def activation_function(x):
        return 1 / (1 + np.exp(-x))
"""
Mean Square Error calculation
"""
def MSE(y, Y):
    np.array(Y)
    #np.reshape = (Y,(2389, 3))
    return np.mean((y-Y)**2)

#Data path for training input
data_path = "total_Bws.xlsx"
df_Train = pd.read_excel(data_path)
df_Train.head()

#Data path for training target

data_path_target = 'BWS_Total_Val10_train_10000g_kopyalayapıstır.xlsx'
df_Target = pd.read_excel(data_path_target)
df_Target.head()

#Data path for test input

data_path_run = 'Adxl_Total_Val10_train_10000g_kopyalayapıstır.xlsx'
df_Run = pd.read_excel(data_path_run)
df_Run.head()

#Data path for normalization of data

data_path_target2 = 'Adxl_Total_Val10_test_1000g.xlsx'#BWS run tarafı için
df_Target2 = pd.read_excel(data_path_target2)
df_Target2.head()

train_features = df_Train
train_targets = df_Target

train_run = df_Run
train_run_targets = df_Target2

# Models' hyperparameter
hidden_nodes = 12
input_nodes = 3
output_nodes = 3

learning_rate = 0.00005

epoch = 1000

weights_input_to_hidden = np.random.normal(0.0, hidden_nodes**-0.5, (input_nodes, hidden_nodes))
weights_hidden_to_output = np.random.normal(0.0, output_nodes**-0.5, (hidden_nodes, output_nodes))

lr = learning_rate

inputs = np.array(train_features, ndmin=2)
targets = np.array(train_targets, ndmin=2)


for i in range(epoch):
    """
    Implementation of FFNN without batch 
    """
    #batch = np.random.choice(train_features.index, size=137)
    for n in range(100): 
        #◙record, target in zip(train_features.iloc[batch].values, train_targets.iloc[batch]):
        hidden_inputs = np.matmul(inputs, weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = activation_function(hidden_inputs) # signals from hidden layer
        
        final_inputs = np.matmul(hidden_outputs,weights_hidden_to_output) # signals into final output layer
        final_outputs = activation_function(final_inputs) # signals from final output layer
        
        # TODO: Output error
        output_errors = targets - final_outputs # Output layer error is the difference between desired target and actual output.
        
        output_grad = final_outputs * (1-final_outputs)
        
        # TODO: Updating weights
        weights_hidden_to_output += lr * np.matmul(hidden_outputs.T,output_grad*output_errors)
        
        hidden_errors = np.matmul(weights_hidden_to_output, output_errors.T).T # errors propagated to the hidden layer
        
        hidden_grad = hidden_outputs * (1 - hidden_outputs)  # hidden layer gradients
        
        weights_input_to_hidden += lr * np.matmul(inputs.T, hidden_grad*hidden_errors) # update hidden-to-output weights with gradient descent step
        
    
    #Run Function
    # Run a forward pass through the network
    run_inputs = np.array(train_run, ndmin=2)
    
    run_hidden_inputs = np.dot(inputs, weights_input_to_hidden) # signals into hidden layer
    run_hidden_outputs = activation_function(run_hidden_inputs) # signals from hidden layer
    
    # TODO: Output layer
    final_outputs = np.dot(run_hidden_outputs ,weights_hidden_to_output ) # signals into final output layer
    final_outputs = activation_function(final_outputs)
    
    # train_targets transposed
    train_loss = MSE(final_outputs, train_run_targets.values)
    #val_loss = MSE(final_outputs, val_targets.values)
    
    sys.stdout.write("\rProgress: " + str(100 * i/float(epoch))[:4] + "% ... Training loss: " + str(train_loss)[:5])
    #"""+  ... Validation loss:  + str(val_loss)[:5]"""