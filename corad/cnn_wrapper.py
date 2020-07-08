import numpy as np
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_std_normalization=True,
    featurewise_center=True
)

train_generator = datagen.flow_from_directory(directory='data/rp_im', color_mode='grey', target_size=(512, 512), class_mode='binary')
validation_generator = datagen.flow_from_directory(directory='data/rp_im', color_mode='grey', target_size=(512, 512), class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800)


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
