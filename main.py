# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 23:35:39 2021

@author: TONG
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import string_utils
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


import tensorflow as tf
import autokeras as ak
from tensorflow.keras.utils import plot_model


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


# df = pd.read_csv('shuffled/data_permutations_both_seq.csv')
# df = df.iloc[0:int(len(df)//2+len(df)//8),:]
# df.drop('Unnamed: 0', axis = 1, inplace = True)

# df2 = pd.read_csv('shuffled/ReversedLightC_dataset.csv')
# df2 = df2.iloc[int(len(df2)//4):int(len(df2)//2),:]
# print(len(df2))
# df2.drop('Unnamed: 0', axis = 1, inplace = True)
# df = df.append(df2)

# df2 = pd.read_csv('shuffled/data_slicing_both_seq.csv')
# df2 = df2.iloc[int(len(df2)//2):,:]
# df2 = df2.iloc[int(len(df2)//2):int(len(df2)//2+len(df2)//4),:]
# print(len(df2))
# df2.drop('Unnamed: 0', axis = 1, inplace = True)
# df = df.append(df2)

# df2 = pd.read_csv('shuffled/orig_random_both.csv')
# df2 = df2.iloc[int(len(df2)//2+len(df2)//4):,:]
# print(len(df2))
# df2.drop('Unnamed: 0', axis = 1, inplace = True)
# df = df.append(df2)
# df.to_csv("shuffled/final.csv",index=False)


df =  pd.read_csv('shuffled/final.csv')
ind = random.sample(range(len(df)),int(len(df)//2))
df = df.iloc[ind,:]


heavy_chain = df.iloc[:,0]
light_chain = df.iloc[:,1]
labels = df.iloc[:,2]


onehot_heavy = aa2onehot(heavy_chain, chain_type = 'heavyChain')
onehot_light = aa2onehot(light_chain, chain_type = 'lightChain')


Y_data = labels

print(onehot_heavy.shape)
print(onehot_light.shape)
train_heavy, test_heavy, train_light, test_light, Y_train, Y_test = train_test_split(onehot_heavy,
                                                                                      onehot_light,
                                                                                      Y_data,
                                                                                      test_size = 0.2,
                                                                                      shuffle = True,
                                                                                      random_state = 11)
# train_heavy = tf.cast(train_heavy, dtype = tf.float32)
# test_heavy = tf.cast(test_heavy, dtype = tf.float32)
# train_light = tf.cast(train_light, dtype = tf.float32)
# test_light = tf.cast(test_light,dtype = tf.float32)

train_heavy = train_heavy.astype(np.float32)
test_heavy  = test_heavy.astype(np.float32)
train_light = train_light.astype(np.float32)
test_light  = test_light.astype(np.float32)

del df,onehot_heavy,onehot_light,Y_data

#train_heavy_1d = np.reshape(train_heavy, (train_heavy.shape[0], train_heavy.shape[1], 1))
#train_light_1d = np.reshape(train_light, (train_light.shape[0], train_light.shape[1], 1))
#test_heavy_1d = np.reshape(test_heavy, (test_heavy.shape[0], test_heavy.shape[1], 1))
#test_light_1d = np.reshape(test_light, (test_light.shape[0], test_light.shape[1], 1))

Y_hot_train = to_categorical(Y_train, num_classes = 2)
Y_hot_test = to_categorical(Y_test, num_classes = 2)

print(train_heavy.shape)
print(test_heavy.shape)
print(train_light.shape)
print(test_light.shape)
print(Y_hot_train.shape)
print(Y_hot_test.shape)

# def aminoAcid_model(heavy_chain, light_chain, num_classes, label_smoothing = 0.05):
#     X_input1 = tf.keras.Input(heavy_chain, name = 'Heavy_chain_data')
#     dense1 = Dense(128, activation = 'relu')(X_input1)
    
#     X_input2 = tf.keras.Input(light_chain, name = 'Light_chain_data')
#     dense2 = Dense(128, activation = 'relu')(X_input2)
    
#     merge = Concatenate()([X_input1, X_input2])
#     dense3 = Dense(128, activation = 'relu')(merge)
#     X_output = Dense(num_classes, activation = 'softmax', name = 'Softmax_layer')(dense3)
                   
#     model = Model(inputs = [X_input1, X_input2], outputs = X_output, name = 'CNN_aminoAcid_model')
#     model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
#                   loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = label_smoothing),
#                   metrics = 'accuracy')
#     return model

# model = aminoAcid_model(train_heavy.shape[1:], train_light.shape[1:], 2)
# training = model.fit(x = [train_heavy, train_light], y = Y_hot_train, batch_size = 8, epochs = 5, 
#                      validation_split = 0.2, shuffle = True, verbose = True)

# testing = model.evaluate(x = [test_heavy, test_light], y = Y_hot_test, verbose = 1)


# prediction = model.predict([test_heavy, test_light])


input_node_hv = ak.Input()
#cnn_node_hv = ak.ConvBlock()(input_node_hv)
#flat_node_hv = ak.SpatialReduction()(cnn_node_hv)
#dense_node_hv = ak.DenseBlock()(flat_node_hv)
dense_node_hv = ak.DenseBlock()(input_node_hv)

input_node_lg = ak.Input()
#cnn_node_lg = ak.ConvBlock()(input_node_lg)
#flat_node_lg = ak.SpatialReduction()(cnn_node_lg)
#dense_node_lg = ak.DenseBlock()(flat_node_lg)
dense_node_lg = ak.DenseBlock()(input_node_lg)

merge_node = ak.Merge(merge_type = 'Concatenate')((dense_node_hv, dense_node_lg))
dense_node_mg = ak.DenseBlock()(merge_node)

output_node = ak.ClassificationHead(num_classes = 2)(dense_node_mg)

clf = ak.AutoModel(
    inputs = (input_node_hv, input_node_lg),
    outputs = output_node,
    project_name = 'Hackaton_keras_Trial03',
    max_trials = 8,
    objective = 'val_loss',
    overwrite = False)

callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
clf.fit(x = (train_heavy, train_light), y = Y_train,
        validation_data = ((test_heavy, test_light), Y_test),
        epochs = 20,
        callbacks = [callback])
print(clf.evaluate((test_heavy, test_light), Y_test))

best_model = clf.export_model()
prediction = best_model.predict((test_heavy, test_light))
dot_img_file = 'Autokeras_bestModel.png'
plot_model(best_model, show_shapes = True, expand_nested = True, to_file = dot_img_file)
try:
    best_model.save('Heavy-Light-model2', save_format = 'tf')
except:
    best_model.save('Heavy-Light-model2.h5')