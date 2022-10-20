import os

import keras
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop

NUM_CLASSES = 10

K = 3
STRIDE = 1
PADDING = ['valid', 'same']


if __name__ == '__main__':
    # Set the Tensorflow verbosity ('0' for DEBUG=all [default], '1' to filter INFO msgs, '2' to filter WARNING msgs, '3' to filter all msgs).
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # Load the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    height, width = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], height, width, 1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], height, width, 1) / 255.0
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Define the model.
    model = Sequential()
    model.add(Conv2D(K, K), S)
    model.add(Dense(NUM_CLASSES, activation='softmax', input_shape=x_train.shape[1:]))
    model.compile(optimizer='adam', loss='category_crossentropy', metrics=['accuracy'])
