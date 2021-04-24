import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

def CNN_aminoAcid_model(heavy_chain, light_chain, num_classes, label_smoothing = 0.05):
    X_input1 = tf.keras.Input(heavy_chain, name = 'Heavy_chain_data')
    dense1 = Dense(256, activation = 'relu')(X_input1)
    
    X_input2 = tf.keras.Input(light_chain, name = 'Light_chain_data')
    dense2 = Dense(256, activation = 'relu')(X_input2)
    
    merge = Concatenate()([X_input1, X_input2])
    dense3 = Dense(256, activation = 'relu')(merge)
    X_output = Dense(num_classes, activaton = 'softmax', name = 'Softmax_layer')(dense3)
                   
    model = Model(inputs = [X_input1, X_input2], outputs = X_output, name = 'CNN_aminoAcid_model')
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
                  loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = label_smoothing),
                  metrics = 'accuracy')
    return model

training = model.fit(x = [train_data1, train_data2], y = train_labels, batch_size = 16, epochs = 50, 
                     validation_split = 0.2, shuffle = True, verbose = True)
testing = model.evaluate(x = [test_data1, test_data2], y = test_labels, verbose = 1)
model.summary()