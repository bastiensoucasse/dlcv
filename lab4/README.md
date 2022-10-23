# PROVOST Iantsa & SOUCASSE Bastien — DLCV Lab 4

## 1. Convolutional Neural Network on MNIST dataset

### 1.2. First CNN

#### Hyperparameters & Results

| Batch Size |  Filters |  Kernel Size  | Stride | Padding |   Loss  | Accuracy |   Time   |
| :--------: | :------: | :-----------: | :----: | :-----: | :-----: | :------: | :------: |
|     32     |    64    |    (3, 3)     |   1    | 'valid' |  0.3152 |  92.43%  | 1411.18s |

<!-- temps abérant, à voir sur ton ordi -->
Results for double plots:
SUMMARY:
    - Loss: 0.3228
    - Accuracy: 0.9154
    - Training Time: 1343.11s

Results for two plots:
SUMMARY:
    - Loss: 0.3107
    - Accuracy: 0.9167
    - Training Time: 1282.95s

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

|    Model    | Accuracy |   Time   |
| :---------: | :------: | :------: |
| best lab3.3 |  97.82%  |  561.08s |
|  first CNN  |  92.43%  | 1411.18s |

<!-- temporary time, to update (also update sentence, as adapted) -->
The CNN model takes two times more time than the lab3.3 best model and provides a 5% lower accuracy. For now, the CNN model doesn't look good. It must be improvable.

<br>

### 1.4. Model improvment

…

<br><br>

## 2. Convolutional Neural Network on CIFAR10 dataset

## 3. Data augmentation

## 4. Transfer learning / fine-tuning on CIFAR10 dataset
