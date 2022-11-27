from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10, mnist

def ten_worst(dataset, y_pred, save: bool = False, path: str = None, probs: bool = False):
    '''
    Find 10 worst classified images, from predictions
        dataset: dataset to work on here, should be cifar10 or mnist

        y_pred: predictions made by the model on x_test (2D array)

        save: True if 10 worst images need to be saved, False otherwise (boolean)

        path: path to where you should save the images (string) here, should be ex_num/keras_or_pytorch/model doesn't need to be defined if save is False 
    '''

    # Load the data.
    _, (x_test, y_test) = dataset.load_data()

    # Determine the number of channels.
    if len(x_test.shape) == 3:
        num_channels = 1
    else:
        num_channels = x_test.shape[3]

    # Retrieve the input size.
    width, height, depth = x_test.shape[2], x_test.shape[1], num_channels

    # Get the real class and predicted class for each image.
    real_classes = y_test
    pred_classes = np.argmax(y_pred, axis=1)

    # Build array of tuple of misclassified images indices,
    # with their predicted class, real class and probability predicted for their real class
    misclassified = [(i, pred_classes[i], real_classes[i], y_pred[i][real_classes[i]]) for i in range(len(real_classes)) if real_classes[i] != pred_classes[i]]

    # Sort it by probabilities and retrieve only the ten worst.
    misclassified.sort(key=lambda a: a[3])
    ranking = misclassified[-10:]

    # Show the ten worst images, their real and predicted classes, along with the probability for their real class.
    x_test = x_test.reshape(x_test.shape[0], height, width, depth) / 255.0
    print('10 WORST CLASSIFIED IMAGES\n')
    for x in range(10, 0, -1):
        i, pc, rc, prob = ranking[x-1]
        print(f'{x}. IMAGE {i}\n    - Predicted category: {pc}\n    - Actual category: {rc}\n')
        if probs:
            print(f'    - Probability: {prob}\n')
        if save:
            Path('ten_worst/%s' % path).mkdir(parents=True, exist_ok=True)

            if depth == 1:
                plt.imsave('ten_worst/%s/%d.png' % (path, x), x_test[i, :].reshape(width, height), cmap=matplotlib.cm.binary)
            else:
                plt.imsave('ten_worst/%s/%d.png' % (path, x), x_test[i, :].reshape(width, height, depth), cmap=matplotlib.cm.binary)

def ten_worst_pytorch(dataset: str, y_pred, save: bool = False, path: str = None):
    '''
    Find 10 worst classified images, from predictions
        dataset: dataset to work on here, should be cifar10 or mnist

        y_pred: predictions made the model on x_test (2D array)

        save: True if 10 worst images need to be saved, False otherwise (boolean)

        path: path to where you should save the images (string) here, should be ex_num/keras_or_pytorch/model doesn't need to be defined if save is False 
    '''

    if dataset == 'mnist':
        dataset = mnist
    if dataset == 'cifar10':
        dataset = cifar10

    ten_worst(dataset, y_pred, save, path)
