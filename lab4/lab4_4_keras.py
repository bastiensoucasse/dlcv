import os
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D, AveragePooling2D, Input)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model

import lab4_utils

MODEL = "no_freeze_data_aug2"

EPOCHS = 20
NUM_CLASSES = 10

BATCH_SIZE = 32


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

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        height_shift_range=0.1,
        width_shift_range=0.1,
    )
    datagen.fit(x_train)

    #Get back the convolutional part of a VGG network trained on ImageNet
    resnet = ResNet50(weights='imagenet', include_top=False)
    # for layer in resnet.layers: layer.trainable=False

    #Create your own input format (here 3x200x200)
    input = Input(shape=(width,height, 3),name = 'image_input')

    #Use the generated model 
    output_resnet = resnet(input)

    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_resnet)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    #Create your own model 
    model = Model(inputs=input, outputs=x)
    model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])

    # datagen = ImageDataGenerator(
    #     horizontal_flip=True,
    #     height_shift_range=0.1,
    #     width_shift_range=0.1,
    # )
    # datagen.fit(x_train)

    # Define the model.

    # # Load ResNet.
    # resnet = ResNet50(include_top=False, # do not load last layer (classifier)
    #                  weights='imagenet',
    #                  input_shape=x_train.shape[1:])

    # output = resnet.layers[-1].output
    # output = Flatten()(output)
    # resnet = Model(resnet.input, outputs=output)
    
    # # Freeze the weights.
    # for layer in resnet.layers:
    #     layer.trainable = False

    # headModel = resnet.output
    # headModel = Flatten(name="flatten")(headModel)
    # headModel = Dense(256, activation="relu")(headModel)
    # headModel = Dropout(0.5)(headModel)
    # headModel = Dense(NUM_CLASSES, activation="softmax")(headModel)
    # model = Model(inputs=resnet.input, outputs=headModel)

    # Create our model using Pre-trained ResNet50.
    # model = Sequential()
    # model.add(resnet)
    # model.add(Dense(512, activation='relu', input_dim=x_train.shape[1:]))
    # model.add(Dropout(0.3))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(NUM_CLASSES, activation='softmax'))
    # # model.summary()
    # model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    # hist = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))  # type: ignore
    hist = model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                     steps_per_epoch=len(x_train) / BATCH_SIZE, 
                     epochs=EPOCHS,
                     validation_data=(x_test, y_test))
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
    plt.savefig('plots/ex4/keras/%s_loss.png' % MODEL)
    plt.clf()

    # Plot the accuracy.
    plt.plot(hist.history['accuracy'], label='Training')
    plt.plot(hist.history['val_accuracy'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epoch')
    plt.savefig('plots/ex4/keras/%s_accuracy.png' % MODEL)
    plt.clf()
    

    # Compute Confusion Matrix.
    y_pred = model.predict(x_test)
    
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('plots/ex4/keras/%s_confusion_matrix.png' % MODEL)   

    # Get 10 worst classified images
    lab4_utils.ten_worst(cifar10, y_pred, True, 'ex4/keras/%s' % MODEL)