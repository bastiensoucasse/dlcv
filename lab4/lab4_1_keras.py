import os
import time

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


EPOCHS = 40
NUM_CLASSES = 10

K = 3
STRIDE = 1
PADDING = ['valid', 'same']
NB_FILTERS = 64


if __name__ == '__main__':
    # Set the Tensorflow verbosity ('0' for DEBUG=all [default], '1' to filter INFO msgs, '2' to filter WARNING msgs, '3' to filter all msgs).
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # Load the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    height, width = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], height, width, 1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], height, width, 1) / 255.0
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Define the model.
    model = Sequential()
    model.add(Conv2D(NB_FILTERS, (K, K), STRIDE, PADDING[0], input_shape=x_train.shape[1:]))
    model.add(Flatten())
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])
    start_time = time.time()
    hist = model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(x_test, y_test))  # type: ignore
    training_time = time.time() - start_time
    loss, accuracy = model.evaluate(x_test, y_test)


    # Display the summary.
    print(f"SUMMARY:\n    - Loss: {loss:.4f}\n    - Accuracy: {accuracy:.4f}\n    - Training Time: {training_time:.2f}s")

    # Plot Training Loss & Validation Accuracy Over Epoch.
    plt.clf()
    plt.plot(hist.history["loss"], label="Training Loss")
    # plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epoch")
    plt.savefig("plots/ex1/keras/first_model_loss_over_epoch.png")
    # plt.ylabel("Loss/Accuracy")
    # plt.title("Training Loss & Validation Accuracy Over Epoch")
    # plt.savefig("plots/ex1/keras/first_model_loss_valacc_over_epoch.png")

    plt.clf()
    plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epoch")
    plt.savefig("plots/ex1/keras/first_model_valacc_over_epoch.png")

    # fig, loss_ax = plt.subplots()

    # color = 'tab:red'
    # loss_ax.set_xlabel('Epoch')
    # loss_ax.set_ylabel('Loss', color=color)
    # loss_ax.plot(hist.history["loss"], label="Training Loss", color=color)
    # loss_ax.tick_params(axis='y', labelcolor=color)

    # acc_ax = loss_ax.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # acc_ax.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    # acc_ax.plot(hist.history["val_accuracy"], label="Validation Accuracy", color=color)
    # acc_ax.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title("Training Loss & Validation Accuracy Over Epoch")
    # plt.savefig("plots/ex1/keras/first_model_loss_valacc_over_epoch_test.png")

    # Compute Confusion Matrix.
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig("plots/ex1/keras/first_model_confusion_matrix.png")
