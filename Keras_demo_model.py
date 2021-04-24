import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

def CNN_aminoAcid_model(input_data, num_classes, label_smoothing = 0.05):
    X_input = tf.keras.Input(input_data, name = 'AminoAcid_data')
    dense1 = Dense(100, activation = 'relu', name = 'First_Linear_Layer')(X_input)
    X_output = Dense(num_classes, activaton = 'softmax', name = 'Softmax_layer')(dense1)
                   
    model = Model(inputs = X_input, outputs = X_output, name = 'CNN_aminoAcid_model')
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
                  loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = label_smoothing),
                  metrics = 'accuracy')
    return model
