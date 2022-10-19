# Neural network with hidden layer for binary classification using Keras

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

EPOCHS = 40
HL_UNITS = [8, 16, 32, 64, 128]
NB_HLU = len(HL_UNITS)

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
    history = []
    acc = []
    
    # TODO: Add inputshape
    for hlu in HL_UNITS:
        model = Sequential()
        model.add(Dense(hlu, activation="sigmoid"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history += [model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(x_test, y_test))]
        loss, accuracy = model.evaluate(x_test, y_test)
        acc += [accuracy]
        print("** %d neurons in hidden layer **" % hlu)
        print("Model Loss: %.2f." % loss)
        print("Model Accuracy: %.2f%%." % (accuracy * 100))


    # Plot Loss
    plt.clf()
    for i in range(NB_HLU):
        plt.plot(history[i].history["loss"], label=HL_UNITS[i])
        # plt.plot(history.history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training loss over Epoch")
    # plt.savefig("plots/lab3_2_keras_%du_loss.png" % HL_UNITS)
    plt.savefig("plots/ex2/lab3_2_keras_hlu_loss.png")

    # Plot Accuracy during training
    plt.clf()
    for i in range(NB_HLU):
        plt.plot(history[i].history["accuracy"], label=HL_UNITS[i])
        # plt.plot(history.history["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training accuracy over Epoch")
    # plt.savefig("plots/lab3_2_keras_%du_accuracy.png" % HL_UNITS)
    plt.savefig("plots/ex2/lab3_2_keras_hlu_accuracy.png")

    # Plot Accuracy over HL units
    plt.clf()
    plt.plot(HL_UNITS, acc)
    plt.xlabel("Neurons in hidden layer")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over number of neurons in hidden layer")
    plt.savefig("plots/ex2/lab3_2_keras_hlu_model_accuracy.png")