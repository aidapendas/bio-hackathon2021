# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 20:18:43 2021

@author: TONG
"""


import pandas as pd
import tensorflow as tf
import numpy as np


# READ DATA
data = pd.read_csv("testset/testset.csv")

############################
# ENCODE DATA
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

heavy_chain = data.iloc[:,0]
light_chain = data.iloc[:,1]

onehot_heavy = aa2onehot(heavy_chain, chain_type = 'heavyChain')
onehot_light = aa2onehot(light_chain, chain_type = 'lightChain')

# PREDICT
onehot_heavy =onehot_heavy.astype(np.float32)
onehot_light =onehot_light.astype(np.float32)
modelfile = 'Heavy-Light-model2'
loaded_model = tf.keras.models.load_model(modelfile)
y_pred =  loaded_model.predict((onehot_heavy, onehot_light))

#############################
true_labels = pd.read_csv("testset/targets.csv")
true_labels = true_labels.iloc[:,0]
predicted_labels = (y_pred>0.5) * 1

c= 0
for i in range(len(y_pred)):
    if int(predicted_labels[i][0]) == int(true_labels[i]):
        c=c+1
accuracy = c/(len(y_pred)) * 100
print("Accuracy is %d" % accuracy)