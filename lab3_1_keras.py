# Single neuron neural network for binary classification using Keras

import time

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

DIGIT = 5
EPOCHS = 40
BATCH_SIZES = [60000, 2048, 1024, 512, 256, 128, 64, 32, 16]
NB_BS = len(BATCH_SIZES)

if __name__ == "__main__":
    # Load the data, flatten x, and set up y for binary classification.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    y_new = np.zeros(y_train.shape)
    y_new[np.where(y_train == 0.0)[0]] = 1
    y_train = y_new
    y_new = np.zeros(y_test.shape)
    y_new[np.where(y_test == 0.0)[0]] = 1
    y_test = y_new

    # Array of history over model version (one history being an array of (training_loss, training_accuracy) over epoch).
    hists = []

    # Array of evaluation score over model version (one evaluation score being (evaluation_loss, evaluation_score)).
    scores = []

    # Array of training time over model version.
    durations = []

    for bs in BATCH_SIZES:
        print(f"\n###\n### BATCH SIZE: {bs}\n###")

        # Define the model and its parameters, train it, and evaluate it.
        model = Sequential()
        model.add(Dense(1, activation="sigmoid", input_shape=(x_test.shape[1],)))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        start_time = time.time()
        history = model.fit(x_train, y_train, batch_size=bs, epochs=EPOCHS, validation_data=(x_test, y_test))  # type: ignore
        training_time = time.time() - start_time
        loss, accuracy = model.evaluate(x_test, y_test)
        hists += [history]
        scores += [(loss, accuracy)]
        durations += [training_time]

        # Display the summary.
        print(f"SUMMARY FOR BATCH SIZE {bs}:\n    - Loss: {loss:.4f}\n    - Accuracy: {accuracy:.4f}\n    - Training Time: {training_time:.2f}s")

    # Plot Training Loss Over Epoch.
    plt.clf()
    for i in range(NB_BS):
        plt.plot(hists[i].history["loss"], label=f"Batch Size: {BATCH_SIZES[i]}")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epoch")
    plt.savefig("plots/ex1/keras/loss_over_epoch.png")

    # Plot Training Accuracy Over Epoch.
    plt.clf()
    for i in range(NB_BS):
        plt.plot(hists[i].history["accuracy"], label=f"Batch Size: {BATCH_SIZES[i]}")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epoch")
    plt.savefig("plots/ex1/keras/accuracy_over_epoch.png")

    # Plot Evaluation Loss Over BS.
    plt.clf()
    plt.plot(BATCH_SIZES, np.array(scores)[:, 0])
    plt.xlabel("Batch Size")
    plt.ylabel("Loss")
    plt.title("Loss over Batch Size")
    plt.savefig("plots/ex1/keras/loss_over_bs.png")

    # Plot Evaluation Accuracy Over BS.
    plt.clf()
    plt.plot(BATCH_SIZES, np.array(scores)[:, 1])
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Batch Size")
    plt.savefig("plots/ex1/keras/accuracy_over_bs.png")

    # Plot Training Time Over BS.
    plt.clf()
    plt.plot(BATCH_SIZES, durations)
    plt.xlabel("Batch Size")
    plt.ylabel("Training Time")
    plt.title("Training Time over Batch Size")
    plt.savefig("plots/ex1/keras/training_time_over_bs.png")
