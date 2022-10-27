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
