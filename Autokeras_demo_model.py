import tensorflow as tf
import autokeras as ak
from tensorflow.keras.utils import plot_model

input_node_hv = ak.Input()
cnn_node_hv = ak.ConvBlock()(input_node_hv)
flat_node_hv = ak.SpatialReduction()(cnn_node_hv)
dense_node_hv = ak.DenseBlock()(flat_node_hv)

input_node_lg = ak.Input()
cnn_node_lg = ak.ConvBlock()(input_node_lg)
flat_node_lg = ak.SpatialReduction()(cnn_node_lg)
dense_node_lg = ak.DenseBlock()(flat_node_lg)

merge_node = ak.Merge(merge_type = 'Concatenate')((dense_node_hv, dense_node_lg))
dense_node_mg = ak.DenseBlock()(merge_node)

output_node = ak.ClassificationHead(loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.05))(dense_node_mg)

clf = ak.AutoModel(
    inputs = (input_node_hv, input_node_lg),
    outputs = output_node,
    project_name = 'Hackaton_keras_Trial01',
    max_trials = 3,
    objective = 'val_loss',
    overwrite = False)

callback = tf.keras.callbacks.Earlystopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
clf.fit(x = (train_heavy, train_light), y = Y_hot_train,
        validation_data = ((test_heavy, test_light), Y_hot_test),
        epochs = 30,
        callbacks = [callback])
print(clf.evaluate((test_heavy, test_light), Y_hot_test))

best_model = clf.export_model()
dot_img_file = 'Autokeras_bestModel.png'
plot_model(best_model, show_shapes = True, expand_nested = True, to_file = dot_img_file)
try:
    best_model.save('Heavy-Light-model', save_format = 'tf')
except:
    best_model.save('Heavy-Light-model.h5')

