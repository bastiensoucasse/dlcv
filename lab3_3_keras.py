# Neural network with hidden layer for mutliclass classification using Keras

import time

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
import keras

EPOCHS = 40
OPTIMIZERS = ["adam", "sgd", "RMSprop"]
NB_OPT = len(OPTIMIZERS)

if __name__ == "__main__":
    # Load Data, flatten x, and set up y for multiclass classification.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    history = []
    durations = []

    # TODO: Add inputshape
    for opt in OPTIMIZERS:

        # Define the model and its parameters, train it, and evaluate it.
        model = Sequential()
        model.add(Dense(64, activation="sigmoid"))
        model.add(Dense(10, activation="softmax"))
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        start_time = time.time()
        history += [model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(x_test, y_test))]
        training_time = time.time() - start_time
        durations += [training_time]
        loss, accuracy = model.evaluate(x_test, y_test)

        # Display the summary.
        print("** %s **" % opt)
        print("Model Loss: %.2f." % loss)
        print("Model Accuracy: %.2f%%." % (accuracy * 100))
        print("Training time: %d", training_time)

    # Plot Training Loss Over Epoch.
    plt.clf()
    for i in range(NB_OPT):
        plt.plot(history[i].history["loss"], label=OPTIMIZERS[i])
        # plt.plot(history.history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training loss over Epoch")
    plt.savefig("plots/lab3_3_keras_opt_loss.png")

    # Plot Training Accuracy Over Epoch.
    plt.clf()
    for i in range(NB_OPT):
        plt.plot(history[i].history["accuracy"], label=OPTIMIZERS[i])
        # plt.plot(history.history["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training accuracy over Epoch")
    plt.savefig("plots/lab3_3_keras_opt_accuracy.png")

    # Plot Training time over Optimizers.
    plt.clf()
    plt.plot(OPTIMIZERS, durations, 'o')
    plt.xlabel("Optimizer")
    plt.ylabel("Duration")
    plt.legend()
    plt.title("Duration over Optimizer")
    plt.savefig("plots/ex2/lab3_2_keras_opt_duration.png")