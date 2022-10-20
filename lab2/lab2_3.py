import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

EPOCHS = 100
LEARNING_RATE = 1
HIDDEN_LAYER_UNITS = 64
DIGITS = 10


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def loss(y, a):
    return -np.sum(y * np.log(a)) / y.shape[0]


def one_hot_encode(y, digits):
    examples = y.shape[0]
    y = y.reshape(1, examples)
    y_new = np.eye(digits)[y.astype('int32')]
    y_new = y_new.T.reshape(digits, examples)
    return y_new


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

    # One-hot encode the output.
    y_train = one_hot_encode(y_train, DIGITS)
    y_test = one_hot_encode(y_test, DIGITS)

    ###
    # TRAINING
    ###

    # Retrieve the batch size.
    m = x_train.shape[1]

    # Shuffle the training data.
    np.random.seed(138)
    shuffle_index = np.random.permutation(m)
    x_train, y_train = x_train[:, shuffle_index], y_train[:, shuffle_index]

    # Initialize input layer.
    n0 = x_train.shape[0]

    # Initialize hidden layer.
    n1 = HIDDEN_LAYER_UNITS
    w1 = .01 * np.random.randn(n1, n0)
    b1 = np.zeros((n1, 1))

    # Initialize output layer.
    n2 = DIGITS
    w2 = .01 * np.random.rand(n2, n1)
    b2 = np.zeros((n2, 1))

    # Train the model.
    c = np.empty((EPOCHS,))
    for e in range(EPOCHS):
        print("Epoch: %d/%d" % (e + 1, EPOCHS))

        z1 = np.dot(w1, x_train) + b1
        a1 = sigmoid(z1)

        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)

        c[e] = np.mean(loss(y_train, a2))
        print("Cost: %f" % c[e])

        dz2 = a2 - y_train
        dw2 = (1 / m) * np.dot(dz2, a1.T)
        db2 = (1 / m) * np.array([np.sum(dz2[i, :]) for i in range(n2)]).reshape((n2, 1))

        dz1 = np.dot(w2.T, dz2) * sigmoid_prime(z1)
        dw1 = (1 / m) * np.dot(dz1, x_train.T)
        db1 = (1 / m) * np.array([np.sum(dz1[i, :]) for i in range(n1)]).reshape((n1, 1))

        w1 -= LEARNING_RATE * dw1
        b1 -= LEARNING_RATE * db1
        w2 -= LEARNING_RATE * dw2
        b2 -= LEARNING_RATE * db2

    # Plot the cost over epoch graph.
    plt.plot(c)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title("Cost Over Epoch")
    plt.savefig("lab2_3.png")

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
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    # Analyse the results.
    for i in range(m):
        c = np.argmax((a2[:, i]))
        a2[:, i] = np.zeros_like(a2[:, i])
        a2[c, i] = 1

    # Compute the accuracy.
    accuracy = np.sum(np.array([np.array_equal(a2[:, i], y_test[:, i]) / m for i in range(m)]))
    print("Model accuracy: %.2f%%" % (accuracy * 100))
