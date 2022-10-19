# Neural network with hidden layer for binary classification using Keras

import time

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

EPOCHS = 40
HL_UNITS = [8, 16, 32, 64, 128]
NB_HLU = len(HL_UNITS)
ACTIVATION_FUN = ["sigmoid", "relu", "tanh"]
NB_AF = len(ACTIVATION_FUN)

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
    # history = []
    # acc = []

    HIST = []

    durations = []
    
    # TODO: Add inputshape
    # for hlu in HL_UNITS:
    for af1 in ACTIVATION_FUN:
        history = []
        for af2 in ACTIVATION_FUN:
            model = Sequential()
            model.add(Dense(HL_UNITS[3], activation=af1))
            model.add(Dense(1, activation=af2))
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            start_time = time.time()
            history += [model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(x_test, y_test))]
            training_time = time.time() - start_time
            durations += [training_time]
            loss, accuracy = model.evaluate(x_test, y_test)
            # acc += [accuracy]
            print("** %s & %s **" % (af1, af2))
            print("Model Loss: %.2f." % loss)
            print("Model Accuracy: %.2f%%." % (accuracy * 100))
            print("Training time: %.0fs." % training_time)
        HIST += [history]


    # Plot Loss
    #for i in range(NB_HLU):
    for i in range(NB_AF):
        plt.clf()
        for j in range(NB_AF):
            # plt.plot(history[i].history["loss"], label=HL_UNITS[i])
            plt.plot(HIST[i][j].history["loss"], label=(ACTIVATION_FUN[i] + " & " + ACTIVATION_FUN[j]))
            # plt.plot(history.history["val_loss"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Training loss over Epoch")
            # plt.savefig("plots/ex2/lab3_2_keras_hlu_loss.png")
            plt.savefig("plots/ex2/lab3_2_keras_af%d_loss.png" % i)

    # Plot Accuracy during training
    
    # for i in range(NB_HLU):
    for i in range(NB_AF):
        plt.clf()
        for j in range(NB_AF):
            # plt.plot(history[i].history["accuracy"], label=HL_UNITS[i])
            plt.plot(HIST[i][j].history["accuracy"], label=(ACTIVATION_FUN[i] + " & " + ACTIVATION_FUN[j]))
            # plt.plot(history.history["val_accuracy"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.title("Training accuracy over Epoch")
            # plt.savefig("plots/ex2/lab3_2_keras_hlu_accuracy.png")
            plt.savefig("plots/ex2/lab3_2_keras_af%d_accuracy.png" % i)


    # Plot Accuracy over HL units
    # plt.clf()
    # plt.plot(HL_UNITS, acc)
    # plt.xlabel("Neurons in hidden layer")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.title("Accuracy over number of neurons in hidden layer")
    # plt.savefig("plots/ex2/lab3_2_keras_hlu_model_accuracy.png")

    # Plot time over HL units/activation function
    plt.clf()

    # plt.plot(HL_UNITS, durations)
    # plt.xlabel("Neurons in hidden layer")

    NP_ACTIVATION = np.array(ACTIVATION_FUN)
    models = np.transpose([np.repeat(NP_ACTIVATION, len(NP_ACTIVATION)), np.tile(NP_ACTIVATION, len(NP_ACTIVATION))])
    models = [model[0] + " & " + model[1] for model in models] 
    plt.figure(figsize=(15, 6))
    plt.plot(models, durations, 'o')
    plt.xlabel("Activation function")

    plt.ylabel("Duration")
    plt.legend()
    # plt.title("Duration over number of neurons in hidden layer")
    # plt.savefig("plots/ex2/lab3_2_keras_hlu_duration.png")
    plt.title("Duration over Activation functions")
    plt.savefig("plots/ex2/lab3_2_keras_af_duration.png")
