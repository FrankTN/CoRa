import numpy as np
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
import corad.ild_cnn.cnn_model as CNN
import corad.ild_cnn.main
import tensorflow as tf

datagen = ImageDataGenerator()



def run_cnn():
    t_g, v_g = prepare_data_loaders()
    batch_x, batch_y = t_g.next()
    CNN.train(batch_x, batch_y, np.ones(len(batch_x)), np.ones(len(batch_y)), params=corad.ild_cnn.main.train_params)

def prepare_data_loaders():
    train_generator = datagen.flow_from_directory(directory='data/rp_im', color_mode='grayscale', target_size=(512, 512),
                                                  class_mode='binary')
    validation_generator = datagen.flow_from_directory(directory='data/rp_im', color_mode='grayscale',
                                                       target_size=(512, 512), class_mode='binary')
    return train_generator, validation_generator

run_cnn()

"""
def prepare_file(file_path, mask_path):
    
    # loading mnist dataset
    (X_train, y_train), (X_val, y_val) = mnist.load_data()

    # adding a singleton dimension and rescale to [0,1]
    X_train = np.asarray(np.expand_dims(X_train, 1)) / float(255)
    X_val = np.asarray(np.expand_dims(X_val, 1)) / float(255)

    # labels to categorical vectors
    uniquelbls = np.unique(y_train)
    nb_classes = uniquelbls.shape[0]
    zbn = np.min(uniquelbls)  # zero based numbering
    y_train = np_utils.to_categorical(y_train - zbn, nb_classes)
    y_val = np_utils.to_categorical(y_val - zbn, nb_classes)

    return (X_train, y_train), (X_val, y_val)
    """
