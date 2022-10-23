# PROVOST Iantsa & SOUCASSE Bastien — DLCV Lab 4

## 1. Convolutional Neural Network on MNIST dataset

### 1.2. First CNN

#### Hyperparameters

| Batch Size |  Filters |  Kernel Size  | Stride | Padding |   Loss  | Accuracy |  Time   |
| :--------: | :------: | :-----------: | :----: | :-----: | :-----: | :------: | :-----: |
|     32     |    64    |    (3, 3)     |   1    | 'valid' |  0.2746 |  92.41%  | 563.05s |

<br>

#### Loss and Accuracy Plot

<img src="plots/ex1/keras/first_model_loss_valacc_over_epoch.png" height="240" />

<br>

#### Confusion matrix

<img src="plots/ex1/keras/first_model_confusion_matrix.png" height="240" />

<br>

<!-- TODO: Comment those results -->
…

<br>

### 1.3. Comparison

Here are the configuration and results of the best model we obtained on lab3.3.

|    Model    | Accuracy |  Time   |
| :---------: | :------: | :-----: |
| best lab3.3 |  97.82%  | 561.08s |
| :---------: | :------: | :-----: |
|  first CNN  |  92.41%% | 563.05s |

Both models took about the same amount of time, but the lab3.3 model provides a 5% higher accuracy.

<br>

### 1.4. Model improvment

…

<br><br>

## 2. Convolutional Neural Network on CIFAR10 dataset

## 3. Data augmentation

## 4. Transfer learning / fine-tuning on CIFAR10 dataset
