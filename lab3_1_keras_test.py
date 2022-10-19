# Single neuron neural network for binary classification using Keras

import time

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

BATCH_SIZES = [60000, 2048, 1024, 512, 256, 128, 64, 32, 16, 8]
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
    scores = []
    durations = []
    for BATCH_SIZE in BATCH_SIZES:
        print("\n###\n### BATCH SIZE %d\n###" % BATCH_SIZE)
        start_time = time.time()
        model = Sequential()
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))
        scores += [model.evaluate(x_test, y_test)]
        durations += [time.time() - start_time]

    # Plot Loss
    plt.clf()
    plt.xscale("log") 
    plt.plot(BATCH_SIZES, np.array(scores)[:, 0])
    plt.xlabel("Batch Size")
    plt.ylabel("Loss")
    plt.title("Loss over Batch Size")
    plt.savefig("plots/lab3_1_keras_bs_cmp_loss.png")

    # Plot Accuracy
    plt.clf()
    plt.xscale("log") 
    plt.plot(BATCH_SIZES, np.array(scores)[:, 1])
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Batch Size")
    plt.savefig("plots/lab3_1_keras_bs_cmp_accuracy.png")

    # Plot Durations
    plt.clf()
    plt.xscale("log") 
    plt.plot(BATCH_SIZES, durations)
    plt.xlabel("Batch Size")
    plt.ylabel("Duration")
    plt.title("Duration over Batch Size")
    plt.savefig("plots/lab3_1_keras_bs_cmp_duration.png")
