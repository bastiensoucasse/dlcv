# Best Model

import time

import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

EPOCHS = 40

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    model = Sequential()
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    start_time = time.time()
    model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(x_test, y_test))
    training_time = time.time() - start_time

    loss, accuracy = model.evaluate(x_test, y_test)

    print(f'SUMMARY:\n    - Loss: {loss:.4f}\n    - Accuracy: {accuracy:.4f}\n    - Training Time: {training_time:.2f}s')
