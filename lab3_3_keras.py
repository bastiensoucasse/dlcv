# Neural network with hidden layer for mutliclass classification using Keras

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
    # Load Data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # Train Model
    history = []

    # TODO: Add inputshape
    for opt in OPTIMIZERS:
        model = Sequential()
        model.add(Dense(64, activation="sigmoid"))
        model.add(Dense(10, activation="softmax"))
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        history += [model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(x_test, y_test))]
        loss, accuracy = model.evaluate(x_test, y_test)
        print("** %s **" % opt)
        print("Model Loss: %.2f." % loss)
        print("Model Accuracy: %.2f%%." % (accuracy * 100))

    # Plot Loss
    plt.clf()
    for i in range(NB_OPT):
        plt.plot(history[i].history["loss"], label=OPTIMIZERS[i])
        # plt.plot(history.history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training loss over Epoch")
    plt.savefig("plots/lab3_3_keras_opt_loss.png")

    # Plot Accuracy
    plt.clf()
    for i in range(NB_OPT):
        plt.plot(history[i].history["accuracy"], label=OPTIMIZERS[i])
        # plt.plot(history.history["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training accuracy over Epoch")
    plt.savefig("plots/lab3_3_keras_opt_accuracy.png")