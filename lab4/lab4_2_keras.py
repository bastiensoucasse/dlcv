import os
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import lab4_utils

MODEL = 'model4'

EPOCHS = 20
NUM_CLASSES = 10

K = 5
P = 2
STRIDE = 1
PADDING = ['valid', 'same']
NB_FILTERS = [32, 64, 128, 256]


if __name__ == '__main__':
    # Set the Tensorflow verbosity ('0' for DEBUG=all [default], '1' to filter INFO msgs, '2' to filter WARNING msgs, '3' to filter all msgs).
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # Load the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    height, width = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], height, width, 3) / 255.0
    x_test = x_test.reshape(x_test.shape[0], height, width, 3) / 255.0
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Define the model.
    model = Sequential()
    model.add(Conv2D(NB_FILTERS[1], K, STRIDE, PADDING[0], input_shape=x_train.shape[1:]))
    model.add(Conv2D(NB_FILTERS[1], K, STRIDE, PADDING[0]))
    model.add(MaxPooling2D(P, 2, PADDING[0]))
    model.add(BatchNormalization())
    model.add(Conv2D(NB_FILTERS[2], K, STRIDE, PADDING[0]))
    model.add(Conv2D(NB_FILTERS[2], K, STRIDE, PADDING[0]))
    model.add(MaxPooling2D(P, 2, PADDING[0]))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])
    start_time = time.time()
    hist = model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(x_test, y_test))  # type: ignore
    training_time = time.time() - start_time

    # Evaluate the model.
    loss, accuracy = model.evaluate(x_test, y_test)

    # Display the summary.
    print(f'SUMMARY:\n    - Loss: {loss:.4f}\n    - Accuracy: {accuracy:.4f}\n    - Training Time: {training_time:.2f}s')


    # Plot the loss.
    plt.plot(hist.history['loss'], label='Training')
    plt.plot(hist.history['val_loss'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epoch')
    plt.savefig('plots/ex2/keras/%s_loss.png' % MODEL)
    plt.clf()

    # Plot the accuracy.
    plt.plot(hist.history['accuracy'], label='Training')
    plt.plot(hist.history['val_accuracy'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epoch')
    plt.savefig('plots/ex2/keras/%s_accuracy.png' % MODEL)
    plt.clf()
    

    # Compute Confusion Matrix.
    y_pred = model.predict(x_test)
    
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('plots/ex2/keras/%s_confusion_matrix.png' % MODEL)   

    # Get 10 worst classified images
    lab4_utils.ten_worst(cifar10, y_pred, True, 'ex2/keras/%s' % MODEL)
