# Notes

## Some shit

DropOut(ratio)
- Ratio = proportion of neurons to turn off
- Turns off some neurons, so the other ones learns.
- Helps fight overfitting

Data normalization/standardization
- Data centered around 0
- (X - mean) / std
- Helps learn better
- BatchNorm: 
    - Normalisation layer
    - To put before or after activation
    - Normalize data during training
    - 2 parameters learned: gamma, beta
    - 2 parameters, not learnable: mean moving average, variance moving average
- Helps fight overfitting


## Convolutional network (CNN)

3 types of layer:
- Convolutional layer: Convolution matrix, but the filter applied to the input is the neurons:
    $$\sigma(k_0 \times p_0 + k_1 \times p_1 + \cdots + k_8 \times p_8 + b)$$
    - stride: shift step
    - padding: 'valid' (doesn't compute on boarders), 'same' (zero padding)
    - `Conv2D((k,k), stride, padding, num_filters)` on 3D data (the filter is actually of size `(k, k, depth)`) -> always produces a 2D image (feature map)
        - `stride=1` & `padding='same'`: `w2=w` & `h2=h`
        - `stride=1` & `padding='valid'`: `w2=w-k+1` & `h2=h-k+1`
        $$w_2 = \left( w - f + 2p \right) / s + 1$$
        with `f` the filter size (`k` above), `p` the padding, and `s` the stride.
        - number of filters = depth of output (1 filter = 1 2D image)
    N.B.: Prefer Conv2D layer w/ s>1 to pooling layer!
- Pooling layer:
    - `MaxPooling2D(shape, …)`: maximum value in array of shape (smaller output!!)
- Fully Connected layer (Dense (keras), Linear (pytorch))

Conv2D, Conv2D, MaxPooling2D, Conv2D, Flatten, TraditionalNN

PyTorch: `Conv2D(input_channels, output_channels, kernel_size)`


To detect overfitting:
- Loss on training set
- Accuracy on test set
on same plot

For confusion matrix:
https://subscription.packtpub.com/book/data/9781838555078/6/ch06lvl1sec34/confusion-matrix

To save array in file and load back:
np.savetxt(filename, array, fmt='%f') 
np.loadtxt(filename)

Prevent overfitting:
https://www.analyticsvidhya.com/blog/2020/09/overfitting-in-cnn-show-to-treat-overfitting-in-convolutional-neural-networks/

Architecture
https://vitalflux.com/different-types-of-cnn-architectures-explained-examples/

# SÉANCE 2

To avoid the number of channels into Linear, use LazyLinear.

## Data augmentation
https://www.kaggle.com/code/mielek/data-augmentation-with-keras-using-cifar-10/notebook
https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
https://medium.com/swlh/
https://colab.research.google.com/drive/1hn284Jk4KDlHbgU1b9sQLJPp5fpyA9G5#scrollTo=dnQWa-QHiWCZ


how-data-augmentation-improves-your-cnn-performance-an-experiment-in-pytorch-and-torchvision-e5fb36d038fb

## Models that work

ILSVRC: ImageNet Large Scale V. Recognition Challenge
1.3M images in ImageNet
1K classes

https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-alexnet-cifar10.ipynb
https://www.kaggle.com/code/vortexkol/alexnet-cnn-architecture-on-tensorflow-beginner

Première étape: utiliser un CNN préentrainé et le réutiliser sur nos données sans rien changer.

Rendu 01/12: les 4 parties en entier
Pour la partie 3: on peut s'inspirer de réseaux existant
Pour la partie 4: pas obligé de mettre la data augmentation, on peut essayer avec ou sans
Parties 1 et 2: corriger le diagnostique d'overfitting
