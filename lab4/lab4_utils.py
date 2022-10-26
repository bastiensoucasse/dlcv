import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def ten_worst(dataset, num_classes, y_pred, save, path=None):
    '''
    Find 10 worst classified images, from predictions
        dataset: dataset to work on here, should be cifar10 or mnist

        num_classes: the number of classes of dataset (int)

        y_pred: predictions made the model on y_test (2D array)

        save: True if 10 worst images need to be saved, False otherwise (boolean)

        path: path to where you should save the images (string) here, should be ex_num/keras_or_pytorch/model doesn't need to be defined if save is False 
    '''

    # Load test data.
    _, (x_test, y_test) = dataset.load_data()
    height, width = x_test.shape[1], x_test.shape[2]
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Get the real class and predicted class for each image.
    real_classes = np.argmax(y_test, axis=1)
    pred_classes = np.argmax(y_pred, axis=1)

    # Build array of tuple of misclassified images indices,
    # with their predicted class, real class and probability predicted for their real class
    misclassified = [(i, pred_classes[i], real_classes[i], y_pred[i][real_classes[i]]) for i in range(y_test.shape[0]) if real_classes[i] != pred_classes[i]]

    # Sort it by probabilities and retrieve only the ten worst.
    misclassified.sort(key=lambda a: a[3])
    ranking = misclassified[-10:]

    # Show the ten worst images and their real and predicted classes.
    x_test = x_test.reshape(x_test.shape[0], height, width, 1) / 255.0
    print('10 WORST CLASSIFIED IMAGES\n')
    for x in range(10, 0, -1):
        i, pc, rc, _ = ranking[x-1]
        print(f'{x}. IMAGE {i}\n    - Predicted category: {pc}\n    - Actual category: {rc}\n')
        if save:
            plt.imsave('ten_worst/%s/%d.png' % (path, x), x_test[i, :].reshape(28, 28), cmap=matplotlib.cm.binary)


def ten_worst():
    return
