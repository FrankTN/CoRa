import numpy as np
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
import corad.ild_cnn.cnn_model as CNN
import tensorflow as tf

train_params = {
     'do' : 0.5,        # Dropout Parameter
     'a'  : 0.3,          # Conv Layers LeakyReLU alpha param [if alpha set to 0 LeakyReLU is equivalent with ReLU]
     'k'  :  4,              # Feature maps k multiplier
     's'  :  1,            # Input Image rescale factor
     'pf' :  1,          # Percentage of the pooling layer: [0,1]
     'pt' :  'Avg',             # Pooling type: Avg, Max
     'fp' : 'proportional',    # Feature maps policy: proportional, static
     'cl' :  5,            # Number of Convolutional Layers
     'opt':  'Adam',          # Optimizer: SGD, Adagrad, Adam
     'obj': 'ce',            # Minimization Objective: mse, ce
     'patience' :  200,       # Patience parameter for early stoping
     'tolerance':  1.005,     # Tolerance parameter for early stoping [default: 1.005, checks if > 0.5%]
     'res_alias':  'res'      # csv results filename alias
}

datagen = ImageDataGenerator()


def run_cnn():
    t_g, v_g = prepare_data_loaders()
    x_train, y_train = t_g.next()
    x_val, y_val = v_g.next()

    # labels to categorical vectors
    uniquelbls = np.unique(y_train)
    nb_classes = uniquelbls.shape[0]
    zbn = np.min(uniquelbls) # zero based numbering
    y_train = np_utils.to_categorical(y_train-zbn, nb_classes)
    y_val = np_utils.to_categorical(y_val-zbn, nb_classes)

    CNN.train(x_train, y_train, x_val, y_val, params=train_params)



def prepare_data_loaders():
    train_generator = datagen.flow_from_directory(directory='data/rp_im', color_mode='grayscale',
                                                  target_size=(512, 512),
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
