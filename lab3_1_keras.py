# Single neuron neural network for binary classification using Keras

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

BATCH_SIZE = 32
EPOCHS = 40

if __name__ == "__main__":
    # Load Data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    y_new = np.zeros(y_train.shape)
    y_new[np.where(y_train == 0.0)[0]] = 1
    y_train = y_new
    y_new = np.zeros(y_test.shape)
    y_new[np.where(y_test == 0.0)[0]] = 1
    y_test = y_new

    # Train Model
    model = Sequential()
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Model Loss: %.2f." % loss)
    print("Model Accuracy: %.2f%%." % (accuracy * 100))

    # Plot Loss
    plt.clf()
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epoch")
    plt.savefig("plots/lab3_1_keras_bs%d_loss.png" % BATCH_SIZE)

    # Plot Accuracy
    plt.clf()
    plt.plot(history.history["accuracy"], label="Training")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epoch")
    plt.savefig("plots/lab3_1_keras_bs%d_accuracy.png" % BATCH_SIZE)
