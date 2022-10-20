import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

EPOCHS = 100
LEARNING_RATE = 1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(y, a):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))


if __name__ == "__main__":
    # Load the data.
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and standadize the data.
    x_train = x_train.reshape(x_train.shape[0], -1).T
    x_test = x_test.reshape(x_test.shape[0], -1).T
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Initialize for binary classification (digit 5).
    y_tmp = np.zeros(y_train.shape)
    y_tmp[np.where(y_train == 5.0)[0]] = 1
    y_train = y_tmp.T
    y_tmp = np.zeros(y_test.shape)
    y_tmp[np.where(y_test == 5.0)[0]] = 1
    y_test = y_tmp.T

    ###
    # TRAINING
    ###

    # Retrieve the batch size.
    m = x_train.shape[1]

    # Shuffle the training data.
    np.random.seed(138)
    shuffle_index = np.random.permutation(m)
    x_train, y_train = x_train[:, shuffle_index], y_train[:, shuffle_index]

    # Initialize the input layer.
    n0 = x_train.shape[0]

    # Initialize the output layer.
    n1 = 1
    w1 = .01 * np.random.randn(n1, n0)
    b1 = np.zeros((n1, 1))

    # Train the model.
    c = np.empty((EPOCHS,))
    for e in range(EPOCHS):
        print("Epoch: %d/%d" % (e + 1, EPOCHS))

        z1 = np.dot(w1, x_train) + b1
        a1 = sigmoid(z1)

        c[e] = np.mean(loss(y_train, a1))
        print("Cost: %f" % c[e])

        dz1 = a1 - y_train
        dw1 = (1 / m) * np.dot(dz1, x_train.T)
        db1 = (1 / m) * np.sum(dz1)

        w1 -= LEARNING_RATE * dw1
        b1 -= LEARNING_RATE * db1

    # Plot the cost over epoch graph.
    plt.plot(c)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title("Cost Over Epoch")
    plt.savefig("lab2_1.png")

    ###
    # TESTING
    ###

    # Retrieve the batch size.
    m = x_test.shape[1]

    # Initialize the input layer.
    n0 = x_test.shape[0]

    # Test the trained model.
    z1 = np.dot(w1, x_test) + b1
    a1 = sigmoid(z1)

    # Analyse the results.
    a1[np.where(a1 < .5)] = 0
    a1[np.where(a1 >= .5)] = 1
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(m):
        if a1[0, i] == 0:
            if (y_test[0, i] == 0):
                tn += 1
            else:
                fn += 1
        else:
            if (y_test[0, i] == 0):
                fp += 1
            else:
                tp += 1
    print("True positives: %d" % tp)
    print("False positives: %d" % fp)
    print("True negatives: %d" % tn)
    print("False negatives: %d" % fn)

    # Compute the accuracy.
    accuracy = (tp + tn) / m
    print("Model accuracy: %.2f%%" % (accuracy * 100))
