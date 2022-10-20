# DLCV - Soucasse Bastien - Provost Iantsa - Lab3

## 1. Single Neuron

### 1.2. Default Batch Size

| Batch Size | Activation | Optimizer |  Loss  | Accuracy |  Time   |
| :--------: | :--------: | :-------: | :----: | :------: | :-----: |
|     32     |  Sigmoid   |   Adam    | 0.0232 |  99.31%  | 465.82s |

<img src="plots/ex1/keras/default_loss_over_epoch.png" height="240" />
<img src="plots/ex1/keras/default_accuracy_over_epoch.png" height="240" />

After 40 epochs, the loss and accuracy tends to stabilize and going further wouldn't be useful.

### 1.3. Different Batch Sizes

| Batch Size |  Loss  | Accuracy |  Time   |
| :--------: | :----: | :------: | :-----: |
|   60000    | 0.3035 |  90.20%  |  4.38s  |
|    2048    | 0.0333 |  99.08%  | 11.50s  |
|    1024    | 0.0244 |  99.25%  | 18.89s  |
|    512     | 0.0222 |  99.25%  | 33.66s  |
|    256     | 0.0214 |  99.30%  | 64.63s  |
|    128     | 0.0227 |  99.24%  | 147.17s |
|     64     | 0.0235 |  99.25%  | 291.96s |
|     32     | 0.0232 |  99.31%  | 465.82s |
|     16     | 0.0243 |  99.29%  | 565.27s |

<img src="plots/ex1/keras/loss_over_bs.png" height="240" />
<img src="plots/ex1/keras/accuracy_over_bs.png" height="240" />
<img src="plots/ex1/keras/training_time_over_bs.png" height="240" />
<img src="plots/ex1/keras/loss_over_epoch.png" height="240" />
<img src="plots/ex1/keras/accuracy_over_epoch.png" height="240" />

We can see that smaller batch sizes give better results but take much longer. A good compromise would be around 32, as the results are nearly the same as smaller values, but the execution time remains acceptable. We will keep this one for our next models.

## 2. A Neural Network with One Hidden Layer

### 2.2. Default Network

|  HLU  |   Activations    | Optimizer | Loss  | Accuracy |  Time  |
| :---: | :--------------: | :-------: | :---: | :------: | :----: |
|  64   | Sigmoid, Sigmoid |   Adam    | 0.02  |  99.77%  | XX.XXs |

<img src="plots/ex2/lab3_2_keras_64u_loss.png" alt="plots/lab3_2_keras_64u_loss.png" height="240" />
<img src="plots/ex2/lab3_2_keras_64u_accuracy.png" alt="plots/lab3_2_keras_64u_accuracy.png" height="240" />

<!-- real loss: 0.0185 -->
…

### 2.3. Different numbers of neurons on hidden layer

| HL Units | Loss  | Accuracy |  Time  |
| :------: | :---: | :------: | :----: |
|    8     | 0.02  |  99.58%  | XX.XXs |
|    16    | 0.02  |  99.72%  | XX.XXs |
|    32    | 0.02  |  99.75%  | XX.XXs |
|   128    | 0.01  |  99.80%  | XX.XXs |

<img src="plots/ex2/lab3_2_keras_hlu_loss.png" alt="plots/lab3_2_keras_hlu_loss.png" height="240" />
<img src="plots/ex2/lab3_2_keras_hlu_accuracy.png" alt="plots/lab3_2_keras_hlu_accuracy.png" height="240" />


**32 neurons**

<!-- real loss: 0.0167 -->
We get the same loss and accuracy than with 64 neurons. But is it faster ? To determine... 
Either way, let's try with 16 neurons to see if the performances are still ok.

**16 neurons**

<!-- real loss: 0.0176 -->
Again, we get really close results on loss and accuracy. We could try even less neurons out of curiosity.

**8 neurons**

<!-- real loss: 0.0171 -->
This time, the accuracy decreased a little bit more, more precisely by about 0.10%. However, the accuracy is still pretty high.

**128 neurons**

Since the accuracy is already very high, it would be overkilled to add more neurons to the hidden layer, at least for this model. Indeed, it would take more time (I guess ?), just to get just results that are just as good.


<!-- real loss: 0.0165 -->
That's the highest accuracy we got, but it's not significantly higher for us to say this model is the best one.

<!-- Maybe put this in the summary part ? -->
**Conclusion:** For this model, the number of neurons in the hidden layer is not a significant hyperparameter. As a consequence, we may chose the one that

### 2.4. Different activation functions

**N.B.:** We won't try softmax on last layer because it's only relevant on multiclass classification.

|  <!--  |   Type   |     HL Units     | Activations | Optimizer |   Loss   | Accuracy | Time |
| :----: | :------: | :--------------: | :---------: | :-------: | :------: | :------: |
| Binary |    64    |    relu, relu    |    Adam     |   0.07    |  99.52%  |  XX.XXs  |
| :----: | :------: | :--------------: |  :-------:  |   :---:   | :------: | :------: |
| Binary |    64    |    tanh, tanh    |    Adam     |   0.02    |  99.73%  |  XX.XXs  | -->  |

|  Type  | HL Units |  Activations  | Optimizer | Loss  | Accuracy |  Time  |
| :----: | :------: | :-----------: | :-------: | :---: | :------: | :----: |
| Binary |    64    | sigmoid, relu |   Adam    | 1.51  |  90.20%  | XX.XXs |
| Binary |    64    | sigmoid, tanh |   Adam    | 0.02  |  99.68%  | XX.XXs |
| Binary |    64    | relu, sigmoid |   Adam    | 0.02  |  99.81%  | XX.XXs |
| Binary |    64    |  relu, relu   |   Adam    | 1.51  |  90.20%  | XX.XXs |
| Binary |    64    |  relu, tanh   |   Adam    | 1.51  |  90.20%  | XX.XXs |
| Binary |    64    | tanh, sigmoid |   Adam    | 0.02  |  99.81%  | XX.XXs |
| Binary |    64    |  tanh, relu   |   Adam    | 1.51  |  90.20%  | XX.XXs |
| Binary |    64    |  tanh, tanh   |   Adam    | 0.03  |  99.58%  | XX.XXs |

<img src="plots/ex2/lab3_2_keras_af0_loss.png" alt="plots/lab3_2_keras_af0_loss.png" height="240" />
<img src="plots/ex2/lab3_2_keras_af1_loss.png" alt="plots/lab3_2_keras_af1_loss.png" height="240" />
<img src="plots/ex2/lab3_2_keras_af2_loss.png" alt="plots/lab3_2_keras_af2_loss.png" height="240" />

<img src="plots/ex2/lab3_2_keras_af0_accuracy.png" alt="plots/lab3_2_keras_af0_accuracy.png" height="240" />
<img src="plots/ex2/lab3_2_keras_af1_accuracy.png" alt="plots/lab3_2_keras_af1_accuracy.png" height="240" />
<img src="plots/ex2/lab3_2_keras_af2_accuracy.png" alt="plots/lab3_2_keras_af2_accuracy.png" height="240" />


## 3. Multiclass Neural Networks

### 3.2. Default Network

|    Type    | HL Units |   Activations    | Optimizer | Loss  | Accuracy |  Time  |
| :--------: | :------: | :--------------: | :-------: | :---: | :------: | :----: |
| Multiclass |    64    | sigmoid, softmax |   Adam    | 0.11  |  97.31%  | XX.XXs | <!-- real loss: 0.1117 --> |

<img src="plots/ex3/lab3_3_keras_adam_loss.png" alt="plots/lab3_3_keras_adam_loss.png" height="240" />
<img src="plots/ex3/lab3_3_keras_adam_accuracy.png" alt="plots/lab3_3_keras_adam_accuracy.png" height="240" />

…

### 3.3. Different optimizers

|    Type    | HL Units |   Activations    | Optimizer | Loss  | Accuracy |  Time  |
| :--------: | :------: | :--------------: | :-------: | :---: | :------: | :----: |
| Multiclass |    64    | sigmoid, softmax |    SGD    | 0.20  |  94.21%  | XX.XXs | <!-- real loss: 0.1971 --> |
| Multiclass |    64    | sigmoid, softmax |  RMSprop  | 0.11  |  97.46%  | XX.XXs | <!-- real loss: 0.1087 --> |

<img src="plots/ex3/lab3_3_keras_opt_loss.png" alt="plots/lab3_3_keras_opt_loss.png" height="240" />
<img src="plots/ex3/lab3_3_keras_opt_accuracy.png" alt="plots/lab3_3_opt_sgd_accuracy.png" height="240" />

…

## 4. Best Network

<!-- TODO (or not) -->

### Summary table






## PyTorch

### Binary Classification: Single Neuron

| Batch Size |  Loss  | Accuracy |  Time   |
| :--------: | :----: | :------: | :-----: |
|   60000    | 0.3027 |  90.97%  | 123.33s |
|    2048    | 0.1008 |  94.53%  | 96.60s  |
|    1024    | 0.0904 |  96.59%  | 98.84s  |
|    512     | 0.0845 |  96.81%  | 100.26s |
|    256     | 0.0822 |  97.31%  | 101.05s |
|    128     | 0.0807 |  97.61%  | 105.33s |
|     64     | 0.0799 |  97.64%  | 112.18s |
|     32     | 0.0790 |  97.74%  | 121.56s |
|     16     | 0.0787 |  97.79%  | 142.27s |

<img src="plots/ex1/pytorch/loss_over_bs.png" height="240" />
<img src="plots/ex1/pytorch/accuracy_over_bs.png" height="240" />
<img src="plots/ex1/pytorch/training_time_over_bs.png" height="240" />
<img src="plots/ex1/pytorch/loss_over_epoch.png" height="240" />
<img src="plots/ex1/pytorch/accuracy_over_epoch.png" height="240" />

As we have seen before, the smaller the batch size, the better the accuracy. But the training time is also growing rapidly. So, in order to keep a good ratio time/effectiveness, a batch size of 32 still seems the better option.

### Binary Classification: Hidden Layer

#### Hidden Layer Units (HLU)

|  HLU  |  Loss  | Accuracy |  Time   |
| :---: | :----: | :------: | :-----: |
|   8   | 0.0091 |  99.75%  | 92.47s  |
|  16   | 0.0024 |  99.97%  | 96.81s  |
|  32   | 0.0002 | 100.00%  | 99.41s  |
|  64   | 0.0000 | 100.00%  | 102.66s |
|  128  | 0.0001 | 100.00%  | 119.12s |

<img src="plots/ex2/pytorch/hlu/loss_over_hlu.png" height="240" />
<img src="plots/ex2/pytorch/hlu/accuracy_over_hlu.png" height="240" />
<img src="plots/ex2/pytorch/hlu/training_time_over_hlu.png" height="240" />
<img src="plots/ex2/pytorch/hlu/loss_over_epoch.png" height="240" />
<img src="plots/ex2/pytorch/hlu/accuracy_over_epoch.png" height="240" />

As expected, the more HLU, the better the accuracy. But the training time is also growing rapidly. So, in order to keep a good ratio time/effectiveness, 64 units still seems the better option.

#### Activation Functions (AF)

<!-- TODO -->

### Multiclass Classification

| Optimizer |  Loss  | Accuracy |  Time   |
| :-------: | :----: | :------: | :-----: |
|   Adam    | 1.4725 |  98.62%  | 98.16s  |
|  RMSprop  | 1.4736 |  98.45%  | 99.47s  |
|    SGD    | 2.0245 |  71.19%  | 101.06s |

<img src="plots/ex3/pytorch/loss_over_opt.png" height="240" />
<img src="plots/ex3/pytorch/accuracy_over_opt.png" height="240" />
<img src="plots/ex3/pytorch/training_time_over_opt.png" height="240" />
<img src="plots/ex3/pytorch/loss_over_epoch.png" height="240" />
<img src="plots/ex3/pytorch/accuracy_over_epoch.png" height="240" />

We can notice that the SGD optimizer doesn't give results as good as the other two on this dataset. Adam and RMSprop are both giving the same results.
