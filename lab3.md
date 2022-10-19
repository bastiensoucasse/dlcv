# DLCV - Soucasse Bastien - Provost Iantsa - Lab3

## 1. Single Neuron

### 1.2. Default Batch Size

|  Type  | Batch Size | Activation | Optimizer | Loss  | Accuracy |  Time  |
| :----: | :--------: | :--------: | :-------: | :---: | :------: | :----: |
| Binary |     32     |  Sigmoid   |   Adam    | 0.03  |  99.15%  | XX.XXs |

<img src="plots/ex1/lab3_1_keras_bs32_loss.png" alt="plots/lab3_1_keras_bs32_loss.png" height="240" />
<img src="plots/ex1/lab3_1_keras_bs32_accuracy.png" alt="plots/lab3_1_keras_bs32_accuracy.png" height="240" />

After 40 epochs, the loss and accuracy tends to stabilize and going further wouldn't be useful.

### 1.3. Different Batch Sizes

 |  Type  | Batch Size | Activation | Optimizer | Loss  | Accuracy |  Time  |
 | :----: | :--------: | :--------: | :-------: | :---: | :------: | :----: |
 | Binary |   60000    |  Sigmoid   |   Adam    | X.XX  |  XX.XX%  | XX.XXs |
 | Binary |    2048    |  Sigmoid   |   Adam    | X.XX  |  XX.XX%  | XX.XXs |
 | Binary |    1024    |  Sigmoid   |   Adam    | X.XX  |  XX.XX%  | XX.XXs |
 | Binary |    512     |  Sigmoid   |   Adam    | X.XX  |  XX.XX%  | XX.XXs |
 | Binary |    256     |  Sigmoid   |   Adam    | X.XX  |  XX.XX%  | XX.XXs |
 | Binary |    128     |  Sigmoid   |   Adam    | X.XX  |  XX.XX%  | XX.XXs |
 | Binary |     64     |  Sigmoid   |   Adam    | X.XX  |  XX.XX%  | XX.XXs |
 | Binary |     32     |  Sigmoid   |   Adam    | X.XX  |  XX.XX%  | XX.XXs |
 | Binary |     16     |  Sigmoid   |   Adam    | X.XX  |  XX.XX%  | XX.XXs |

<img src="plots/ex1/lab3_1_keras_bs_cmp_loss.png" alt="plots/lab3_1_keras_bs_cmp_loss.png" height="240" />
<img src="plots/ex1/lab3_1_keras_bs_cmp_accuracy.png" alt="plots/lab3_1_keras_bs_cmp_accuracy.png" height="240" />
<img src="plots/ex1/lab3_1_keras_bs_cmp_duration.png" alt="plots/lab3_1_keras_bs_cmp_duration.png" height="240" />

We can see that smaller batch sizes give better results but take much longer. A good compromise would be around 32, as the results are nearly the same as smaller values, but the execution time remains acceptable. We will keep this one for our next models.

## 2. A Neural Network with One Hidden Layer

### 2.2. Default Network

|  Type  | HL Units |   Activations    | Optimizer | Loss  | Accuracy |  Time  |
| :----: | :------: | :--------------: | :-------: | :---: | :------: | :----: |
| Binary |    64    | Sigmoid, Sigmoid |   Adam    | 0.02  |  99.77%  | XX.XXs |

<img src="plots/ex2/lab3_2_keras_64u_loss.png" alt="plots/lab3_2_keras_64u_loss.png" height="240" />
<img src="plots/ex2/lab3_2_keras_64u_accuracy.png" alt="plots/lab3_2_keras_64u_accuracy.png" height="240" />

<!-- real loss: 0.0185 -->
…

### 2.3. Different numbers of neurons on hidden layer

|  Type  | HL Units |   Activations    | Optimizer | Loss  | Accuracy |  Time  |
| :----: | :------: | :--------------: | :-------: | :---: | :------: | :----: |
| Binary |    8     | Sigmoid, Sigmoid |   Adam    | 0.02  |  99.58%  | XX.XXs |
| Binary |    16    | Sigmoid, Sigmoid |   Adam    | 0.02  |  99.72%  | XX.XXs |
| Binary |    32    | Sigmoid, Sigmoid |   Adam    | 0.02  |  99.75%  | XX.XXs |
| Binary |   128    | Sigmoid, Sigmoid |   Adam    | 0.01  |  99.80%  | XX.XXs |

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






## PYTORCH

### Binary Classification: Single Neuron

|  Type  | Batch Size | Activation | Optimizer |  Loss  | Accuracy |  Time   |
| :----: | :--------: | :--------: | :-------: | :----: | :------: | :-----: |
| Binary |   60000    |  Sigmoid   |   Adam    | 0.3021 |  90.97%  | 119.37s |
| Binary |    2048    |  Sigmoid   |   Adam    | 0.1011 |  94.52%  | 83.70s  |
| Binary |    1024    |  Sigmoid   |   Adam    | 0.0904 |  96.59%  | 86.38s  |
| Binary |    512     |  Sigmoid   |   Adam    | 0.0844 |  96.80%  | 85.34s  |
| Binary |    256     |  Sigmoid   |   Adam    | 0.0822 |  97.35%  | 87.18s  |
| Binary |    128     |  Sigmoid   |   Adam    | 0.0809 |  97.56%  | 90.80s  |
| Binary |     64     |  Sigmoid   |   Adam    | 0.0797 |  97.66%  | 100.32s |
| Binary |     32     |  Sigmoid   |   Adam    | 0.0814 |  97.57%  | 112.29s |
| Binary |     16     |  Sigmoid   |   Adam    | 0.0800 |  97.78%  | 122.05s |

<img src="plots/ex1/pytorch/loss_over_bs.png" height="240" />
<img src="plots/ex1/pytorch/accuracy_over_bs.png" height="240" />
<img src="plots/ex1/pytorch/training_time_over_bs.png" height="240" />
<img src="plots/ex1/pytorch/loss_over_epoch.png" height="240" />
<img src="plots/ex1/pytorch/accuracy_over_epoch.png" height="240" />

…

### Binary Classification: Hidden Layer

#### Hidden Layer Units (HLU)

|  Type  | HL Units |   Activations    | Optimizer |  Loss  | Accuracy |  Time   |
| :----: | :------: | :--------------: | :-------: | :----: | :------: | :-----: |
| Binary |    8     | Sigmoid, Sigmoid |   Adam    | 0.0091 |  99.75%  | 92.47s  |
| Binary |    16    | Sigmoid, Sigmoid |   Adam    | 0.0024 |  99.97%  | 96.81s  |
| Binary |    32    | Sigmoid, Sigmoid |   Adam    | 0.0002 | 100.00%  | 99.41s  |
| Binary |    64    | Sigmoid, Sigmoid |   Adam    | 0.0000 | 100.00%  | 102.66s |
| Binary |   128    | Sigmoid, Sigmoid |   Adam    | 0.0001 | 100.00%  | 119.12s |

<img src="plots/ex2/pytorch/hlu/loss_over_hlu.png" height="240" />
<img src="plots/ex2/pytorch/hlu/accuracy_over_hlu.png" height="240" />
<img src="plots/ex2/pytorch/hlu/training_time_over_hlu.png" height="240" />
<img src="plots/ex2/pytorch/hlu/loss_over_epoch.png" height="240" />
<img src="plots/ex2/pytorch/hlu/accuracy_over_epoch.png" height="240" />

…

#### Activation Functions (AF)

<!-- TODO -->

### Multiclass Classification

<!-- TODO -->
