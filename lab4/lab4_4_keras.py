import os
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import AveragePooling2D, BatchNormalization, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.applications.resnet50 import ResNet50

import lab4_utils

MODEL = "MyResNetDA"

EPOCHS = 20
NUM_CLASSES = 10

BATCH_SIZE = 32


if __name__ == '__main__':
    # Set the Tensorflow verbosity ('0' for DEBUG=all [default], '1' to filter INFO msgs, '2' to filter WARNING msgs, '3' to filter all msgs).
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # Load the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Initialize Data Augmentation.
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        height_shift_range=0.1,
        width_shift_range=0.1,
    )
    datagen.fit(x_train)

    # Define the ResNet model.
    resnet = ResNet50(weights='imagenet', include_top=False)

    # Define the model.
    input = Input(shape=x_train.shape[1:], name='input')
    resnet_output = resnet(input)
    avg_pool = GlobalAveragePooling2D(name='avg_pool')(resnet_output)
    predictions = Dense(NUM_CLASSES, activation='softmax', name='predictions')(avg_pool)
    model = Model(input, predictions)
    model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Model: {MODEL}.")

    # Train the model.
    start_time = time.time()
    # hist = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))
    hist = model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), steps_per_epoch=len(x_train) / BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))
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
    plt.savefig('plots/ex4/keras/%s/loss.png' % MODEL)
    plt.clf()

    # Plot the accuracy.
    plt.plot(hist.history['accuracy'], label='Training')
    plt.plot(hist.history['val_accuracy'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epoch')
    plt.savefig('plots/ex4/keras/%s/accuracy.png' % MODEL)
    plt.clf()

    # Compute Confusion Matrix.
    y_pred = model.predict(x_test)
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('plots/ex4/keras/%s/confusion_matrix.png' % MODEL)   

    # Get 10 worst classified images
    lab4_utils.ten_worst(cifar10, y_pred, True, 'ex4/keras/%s' % MODEL, True)
