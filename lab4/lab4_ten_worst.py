import os
import time

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10, mnist


# Find 10 worst classified images, from predictions file name
def ten_worst(dataset, num_classes, filename):
    
    # Load predictions and real classes.
    _, (x_test, y_test) = mnist.load_data()
    height, width = x_test.shape[1], x_test.shape[2]
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_pred = np.loadtxt(filename)

    # Get the real class and predicted class for each image.
    real_classes = np.argmax(y_test, axis=1)
    pred_classes = np.argmax(y_pred, axis=1)

    # Build array of tuple of misclassified images indices, 
    # with their predicted class, real class and probability predicted for their real class
    misclassified = [(i, pred_classes[i], real_classes[i], y_pred[i][real_classes[i]])
                      for i in range(y_test.shape[0]) if real_classes[i] != pred_classes[i]]
    
    # Sort it by probabilities and retrieve only the ten worst.
    misclassified.sort(key=lambda a: a[3])
    ranking = misclassified[-10:]

    # Show the ten worst images.
    x_test = x_test.reshape(x_test.shape[0], height, width, 1) / 255.0
    print("10 WORST CLASSIFIED IMAGES\n")
    for x in range(10, 0, -1):
        i, pc, rc, _ = ranking[x-1]
        print(f"{x}. IMAGE {i}\n    - Predicted category: {pc}\n    - Actual category: {rc}\n")
        # plt.imshow(x_test[i,:].reshape(28,28), cmap = matplotlib.cm.binary)
        # plt.axis("off")
        # plt.title("Rank %d" % x)
    
    
ten_worst(mnist, 10, 'preds/ex1/keras/model_1_pred.txt')
