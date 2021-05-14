# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 23:35:39 2021

@author: TONG, ERIC TONG
"""
import random
import numpy as np
import pandas as pd
import autokeras as ak
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model

from sklearn.model_selection import train_test_split

def aa2int(seq : str) -> list:
    return [dic_aa2int[i] for i in seq]

def aa2onehot(list_of_sequences, chain_type = None):
    if chain_type == 'heavyChain':
        seq_len = 150
    elif chain_type == 'lightChain':
        seq_len = 130
    else:
        print('Problem with chain type...')
        return
    
    n_amino = 20
    onehot_data = np.zeros((len(list_of_sequences), seq_len, n_amino))
    for index, seq in enumerate(list_of_sequences):  
        output = np.zeros((seq_len, n_amino))
        c = 0
        for i in aa2int(seq):
            temp = np.zeros((n_amino))
            if i == 0:
                output[c] = temp
            else:
                temp[i-1] = 1
                output[c] = temp
            c = c+1
        
        onehot_data[index] = output
    onehot_data_reshape = np.reshape(onehot_data, (onehot_data.shape[0], onehot_data.shape[1]*onehot_data.shape[2]))
    return onehot_data_reshape

dic_aa2int = {'A' : 1,
              'R' : 2,
              'N' : 3,
              'D' : 4,
              'C' : 5,
              'Q' : 6,
              'E' : 7,
              'G' : 8,
              'H' : 9,
              'I' : 10,
              'L' : 11,
              'K' : 12,
              'M' : 13,
              'F' : 14,
              'P' : 15,
              'S' : 16,
              'T' : 17,
              'W' : 18,
              'Y' : 19,
              'V' : 20,
              'X' : 0,
              '-' : 0,
              '*' : 0,
              '?' : 0}

df =  pd.read_csv('trainset_shuffled/final.csv')
ind = random.sample(range(len(df)),int(len(df)//2))
df = df.iloc[ind,:]

heavy_chain = df.iloc[:,0]
light_chain = df.iloc[:,1]
labels = df.iloc[:,2]

onehot_heavy = aa2onehot(heavy_chain, chain_type = 'heavyChain')
onehot_light = aa2onehot(light_chain, chain_type = 'lightChain')
Y_data = labels

train_heavy, test_heavy, train_light, test_light, Y_train, Y_test = train_test_split(onehot_heavy,
                                                                                     onehot_light,
                                                                                     Y_data,
                                                                                     test_size = 0.2,
                                                                                     shuffle = True,
                                                                                     random_state = 11)

train_heavy = train_heavy.astype(np.float32)
test_heavy  = test_heavy.astype(np.float32)
train_light = train_light.astype(np.float32)
test_light  = test_light.astype(np.float32)
Y_hot_train = to_categorical(Y_train, num_classes = 2)
Y_hot_test = to_categorical(Y_test, num_classes = 2)
print(train_heavy.shape)
print(test_heavy.shape)
print(train_light.shape)
print(test_light.shape)
print(Y_hot_train.shape)
print(Y_hot_test.shape)

del df,onehot_heavy,onehot_light,Y_data # save memory by deleting data that is not used for the network

input_node_hv = ak.Input()
dense_node_hv = ak.DenseBlock()(input_node_hv)

input_node_lg = ak.Input()
dense_node_lg = ak.DenseBlock()(input_node_lg)

merge_node = ak.Merge(merge_type = 'Concatenate')((dense_node_hv, dense_node_lg))
dense_node_mg = ak.DenseBlock()(merge_node)

output_node = ak.ClassificationHead(num_classes = 2)(dense_node_mg)

clf = ak.AutoModel(
    inputs = (input_node_hv, input_node_lg),
    outputs = output_node,
    project_name = 'Hackaton_keras_Trial01',
    max_trials = 20,
    objective = 'val_loss',
    overwrite = False)

callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
clf.fit(x = (train_heavy, train_light), y = Y_train,
        validation_data = ((test_heavy, test_light), Y_test),
        epochs = 30,
        callbacks = [callback])
print(clf.evaluate((test_heavy, test_light), Y_test))
best_model = clf.export_model()

## Local Testing ##
true_labels = pd.read_csv("testset/targets.csv")
true_labels = true_labels.iloc[:,0]

data = pd.read_csv("testset/testset.csv")
heavy_chain = data.iloc[:,0]
light_chain = data.iloc[:,1]

onehot_heavy = aa2onehot(heavy_chain, chain_type = 'heavyChain')
onehot_light = aa2onehot(light_chain, chain_type = 'lightChain')

onehot_heavy = onehot_heavy.astype(np.float32)
onehot_light = onehot_light.astype(np.float32)
y_pred =  best_model.evaluate((onehot_heavy, onehot_light), true_labels)
print("Accuracy is %d" % y_pred[1])
