# PROVOST Iantsa & SOUCASSE Bastien — DLCV Lab 4

- Development
    - **Apple MacBook Air (2017)**
        - Intel Core i5 Chip: 1.8GHz dual-core Intel Core i5, Turbo Boost up to 2.9GHz, with 3MB shared L3 cache.
    - **Apple MacBook Air (2020)**
        - Apple M1 Chip: 8-core CPU with 4 perform­ance cores and 4 efficiency cores, 7-core GPU, and 16-core Neural Engine.
- Testing
    - **CREMI (201)**
        - Intel Xeon W-1290 12-core CPU, and RTX 3060 12Go GPU.

<br /><br />

# Keras

## 1. Convolutional Neural Network on MNIST Dataset

### 1.2. First CNN

*N.B.:* After running the program with 40 epochs and realizing it takes a lot of time, we decided to lower the number of epochs to 20. Indeed, the results seem to stabilize around that number.

#### Model Summary

|   ID   |  Loss  | Accuracy | Training Time |
| :----: | :----: | :------: | :-----------: |
| model1 | 0.2940 |  91.96%  |    46.96s     |

- Conv2D: 32, 3, 1, 'valid'.
- Flatten.
- Dense: 10, 'softmax'.

<br />

#### Loss and Accuracy Plots

<img src="plots/ex1/keras/model1_loss.png" height="240" />
<img src="plots/ex1/keras/model1_accuracy.png" height="240" />

<br />

The plots for training data seem normal, but the validation data give don't follow: the model is overfitting. 

<br />

#### Confusion Matrix

<img src="plots/ex1/keras/model1_confusion_matrix.png" height="400" />

<br />

The diagonal is where there are the higher numbers, which is a good thing since it represents the true positives. When it comes to misclassified images, we can observe that the most misclassified digits are:
- 5 as 3 (34)
- 8 as 5 (37)
- 9 as 4 (42)
- 9 as 7 (42)

<br />

#### 10 Worst Classified Images

First of all, let's define what we mean by "badly classified" images. Here, we consider an image badly classified if:
- it is misclassified
- the probability predicted by the model that it's their actual category is low

As a consequence, we decided to gather all the misclassified images and selected the ones that had the ten lowest predicted probability for their actual class.

<br />

| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                      Image                      |
| :---: | :--------: | :--------: | :-------: | :---------------------------------------------: |
|  10   |    1727    |     7      |     3     | <img src="ten_worst/ex1/keras/model1/10.png" /> |
|   9   |    6511    |     5      |     3     | <img src="ten_worst/ex1/keras/model1/9.png" />  |
|   8   |    4910    |     4      |     9     | <img src="ten_worst/ex1/keras/model1/8.png" />  |
|   7   |    565     |     9      |     4     | <img src="ten_worst/ex1/keras/model1/7.png" />  |
|   6   |    5874    |     3      |     5     | <img src="ten_worst/ex1/keras/model1/6.png" />  |
|   5   |    7786    |     7      |     9     | <img src="ten_worst/ex1/keras/model1/5.png" />  |
|   4   |    8297    |     5      |     8     | <img src="ten_worst/ex1/keras/model1/4.png" />  |
|   3   |    7689    |     5      |     8     | <img src="ten_worst/ex1/keras/model1/3.png" />  |
|   2   |    3862    |     3      |     2     | <img src="ten_worst/ex1/keras/model1/2.png" />  |
|   1   |    2371    |     9      |     4     | <img src="ten_worst/ex1/keras/model1/1.png" />  |

<br />

Note that this ranking is for an arbitrary run.

In this ranking, we can notice that there are the most confusions between:
- 5 and 3 (2)
- 5 and 8 (2)
- 4 and 9 (3)

Looking back at the confusion matrix, we can see that those 3 confusions all appear in the most misclassified digits list, at least in one way (the two ways being x misclassified as y, and y miscalssified as x).

<br />

### 1.3. Comparison

|    Model    | Accuracy |  Time  |
| :---------: | :------: | :----: |
| best lab3.3 |  97.59%  | 42.64s |
|   model1    |  91.96%  | 46.96s |

The CNN model takes a little bit more time (5s) than the lab3.3 best model and provides an about 5% lower accuracy. For now, the CNN model is not better but it must be improvable.

<br />

### 1.4. Model Improvement

### 1.4.1. A New Architecture

Let's use an architecture more complex given in class.

#### Model Summary

|   ID   |  Loss  | Accuracy | Training Time |
| :----: | :----: | :------: | :-----------: |
| model2 | 0.5195 |  97.81%  |    144.61s    |

- Conv2D: 32, 3, 1, 'valid'.
- Conv2D: 64, 3, 1, 'valid'.
- MaxPooling: 2, 1, 'valid'
- Conv2D: 128, 3, 1, 'valid'.
- Flatten.
- Dense: 10, 'softmax'.

This model's accuracy is much better, it even reaches the lab3.3 best model accuracy. However, the training time is way longer (about 3 times), but it remains reasonable.

<br />

#### Loss and Accuracy Plots

<img src="plots/ex1/keras/model2_loss.png" height="240" />
<img src="plots/ex1/keras/model2_accuracy.png" height="240" />

<br />

Our model is definitely **overfitting** but later than model1. Indeed, even though we got a very good accuracy and the training loss is decreasing as expected, the validation loss is increasing.

<br />

#### Confusion Matrix

<img src="plots/ex1/keras/model2_confusion_matrix.png" height="400" />

<br />

Just like the first model, the diagonal is where there are the higher numbers. Moreover there are very few misclassified images (which is logical since the accuracy is higher). When it comes to misclassified images, we can observe that the most misclassified digits are:
- 6 as 0 (10)
- 8 as 7 (10)
- 5 as 3 (11) (also noticed in first model)
- 7 as 2 (12)
- 9 as 7 (19)

There are less misclassified images but more categories.

<br />

#### 10 Worst Classified Images

*N.B.*: To know what is meant by "10 worst classified images", see same section in **1.2**.

<br />

| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                      Image                      |
| :---: | :--------: | :--------: | :-------: | :---------------------------------------------: |
|  10   |    7813    |     8      |     9     | <img src="ten_worst/ex1/keras/model2/10.png" /> |
|   9   |    2135    |     1      |     6     | <img src="ten_worst/ex1/keras/model2/9.png" />  |
|   8   |    2298    |     0      |     8     | <img src="ten_worst/ex1/keras/model2/8.png" />  |
|   7   |    290     |     5      |     8     | <img src="ten_worst/ex1/keras/model2/7.png" />  |
|   6   |    5936    |     9      |     4     | <img src="ten_worst/ex1/keras/model2/6.png" />  |
|   5   |    4838    |     5      |     6     | <img src="ten_worst/ex1/keras/model2/5.png" />  |
|   4   |    9982    |     6      |     5     | <img src="ten_worst/ex1/keras/model2/4.png" />  |
|   3   |    2770    |     7      |     3     | <img src="ten_worst/ex1/keras/model2/3.png" />  |
|   2   |    7886    |     4      |     2     | <img src="ten_worst/ex1/keras/model2/2.png" />  |
|   1   |    3794    |     3      |     8     | <img src="ten_worst/ex1/keras/model2/1.png" />  |

<br />

First of all, none of the images in this ranking appear in the first model ranking.
Then, we can only observe 1 confusion in both ways between 5 and 6, which was not in the first model ranking. Also this confusion doesn't appear in the most misclassified images. But, they look "harder to recognize" (such as the 7th, 8th and 9th) than the ones in the first model.

<br />

### 1.4.2. Fighting Against Overfitting

This time, let's build a model with data normalization, to prevent overfitting. And then, try and improve its accuracy.

#### Model Summaries

Legend:
- Conv2D: num_filters, kernel_size, stride, padding.
- MaxPooling: pool_size, stride, padding.

|   ID   | Architecture                                                                                                                                                                                                         |  Loss  | Accuracy | Training time |
| :----: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: | :------: | :-----------: |
| model3 | - Conv2D: 32, `5`, 1, 'valid' <br /> - Conv2D: 64, 5, 1, 'valid' <br /> - `BatchNorm` <br /> - MaxPooling: 2, 1, 'valid' <br /> - Conv2D: 128, 5, 1, 'valid' <br /> - Flatten <br /> - Dense: 10, 'softmax'          | 0.2406 |  98.12%  |    143.87s    |
| model4 | - Conv2D: 32, `5`, 1, 'valid' <br /> - Conv2D: 64, 5, 1, 'valid' <br /> - `BatchNorm` <br /> - MaxPooling: 2, `2`, 'valid' <br /> - Conv2D: 128, 5, 1, 'valid' <br /> - Flatten <br /> - Dense: 10, 'softmax'        | 0.2260 |  98.13%  |    113.17s    |
| model5 | - Conv2D: `64`, `5`, 1, 'valid' <br /> - Conv2D: `128`, 5, 1, 'valid' <br /> - `BatchNorm` <br /> - MaxPooling: 2, `2`, 'valid' <br /> - Conv2D: `256`, 5, 1, 'valid' <br /> - Flatten <br /> - Dense: 10, 'softmax' | 0.3216 |  98.04%  |    211.00s    |

Those 3 new models provide rather identical accuracies, that are slightly better than the model2 one. The training time allows to decide between them. Indeed, even if these information are not sufficient to choose a model, model4 seems to be the best model so far.

#### Loss, Accuracy Plots and Confusion Matrices

<br />

|   ID   |                         Loss Plot                          |                         Accuracy Plot                          |                            Confusion Matrix                            |
| :----: | :--------------------------------------------------------: | :------------------------------------------------------------: | :--------------------------------------------------------------------: |
| model3 | <img src="plots/ex1/keras/model3_loss.png" height="150" /> | <img src="plots/ex1/keras/model3_accuracy.png" height="150" /> | <img src="plots/ex1/keras/model3_confusion_matrix.png" height="150" /> |
| model4 | <img src="plots/ex1/keras/model4_loss.png" height="150" /> | <img src="plots/ex1/keras/model4_accuracy.png" height="150" /> | <img src="plots/ex1/keras/model4_confusion_matrix.png" height="150" /> |
| model5 | <img src="plots/ex1/keras/model5_loss.png" height="150" /> | <img src="plots/ex1/keras/model5_accuracy.png" height="150" /> | <img src="plots/ex1/keras/model5_confusion_matrix.png" height="150" /> |

Whether it is about loss or accuracy, for all 3 models, we can observe some overfitting since the training values are improving and not the validation ones.
Nevertheless, the scale is small so even if it may look huge, they actually all have:
- about 0.2-0.25 delta for the loss
- +/- 0.2 delta for the accuracy

They all look quite equivalent. More importantly, they show better results than model2. 

<br />

#### 10 Worst Classified Images

|       |   model3   |            |           |                                                 |   model4   |            |           |                                                 |   model5   |            |           |                                                 |
| :---: | :--------: | :--------: | :-------: | :---------------------------------------------: | :--------: | :--------: | :-------: | :---------------------------------------------: | :--------: | :--------: | :-------: | :---------------------------------------------: |
| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                      Image                      | Image Idx. | Pred. Cat. | Act. Cat. |                      Image                      | Image Idx. | Pred. Cat. | Act. Cat. |                      Image                      |
|  10   |    4196    |     9      |     5     | <img src="ten_worst/ex1/keras/model3/10.png" /> |    3559    |     5      |     8     | <img src="ten_worst/ex1/keras/model4/10.png" /> |    9614    |     5      |     3     | <img src="ten_worst/ex1/keras/model5/10.png" /> |
|   9   |    924     |     7      |     2     | <img src="ten_worst/ex1/keras/model3/9.png" />  |    9904    |     8      |     2     | <img src="ten_worst/ex1/keras/model4/9.png" />  |    4534    |     7      |     9     | <img src="ten_worst/ex1/keras/model5/9.png" />  |
|   8   |    6157    |     5      |     9     | <img src="ten_worst/ex1/keras/model3/8.png" />  |    9698    |     5      |     6     | <img src="ten_worst/ex1/keras/model4/8.png" />  |    6624    |     5      |     3     | <img src="ten_worst/ex1/keras/model5/8.png" />  |
|   7   |    6166    |     3      |     9     | <img src="ten_worst/ex1/keras/model3/7.png" />  |    1101    |     3      |     8     | <img src="ten_worst/ex1/keras/model4/7.png" />  |    5922    |     3      |     5     | <img src="ten_worst/ex1/keras/model5/7.png" />  |
|   6   |    9645    |     7      |     1     | <img src="ten_worst/ex1/keras/model3/6.png" />  |    9331    |     3      |     5     | <img src="ten_worst/ex1/keras/model4/6.png" />  |    1686    |     6      |     8     | <img src="ten_worst/ex1/keras/model5/6.png" />  |
|   5   |    5176    |     4      |     8     | <img src="ten_worst/ex1/keras/model3/5.png" />  |    5265    |     4      |     6     | <img src="ten_worst/ex1/keras/model4/5.png" />  |    4783    |     9      |     4     | <img src="ten_worst/ex1/keras/model5/5.png" />  |
|   4   |    1138    |     1      |     2     | <img src="ten_worst/ex1/keras/model3/4.png" />  |    6651    |     8      |     0     | <img src="ten_worst/ex1/keras/model4/4.png" />  |    2406    |     4      |     9     | <img src="ten_worst/ex1/keras/model5/4.png" />  |
|   3   |    543     |     7      |     8     | <img src="ten_worst/ex1/keras/model3/3.png" />  |    6391    |     4      |     2     | <img src="ten_worst/ex1/keras/model4/3.png" />  |    3941    |     6      |     4     | <img src="ten_worst/ex1/keras/model5/3.png" />  |
|   2   |    4256    |     2      |     3     | <img src="ten_worst/ex1/keras/model3/2.png" />  |    5745    |     1      |     7     | <img src="ten_worst/ex1/keras/model4/2.png" />  |    2189    |     8      |     9     | <img src="ten_worst/ex1/keras/model5/2.png" />  |
|   1   |    2369    |     3      |     5     | <img src="ten_worst/ex1/keras/model3/1.png" />  |    9638    |     7      |     9     | <img src="ten_worst/ex1/keras/model4/1.png" />  |    3951    |     7      |     8     | <img src="ten_worst/ex1/keras/model5/1.png" />  |

<br /><br />

## 2. Convolutional Neural Network on CIFAR10 Dataset

### 2.2. First CNN
### Model Summary

|   ID   |  Loss  | Accuracy | Training Time |
| :----: | :----: | :------: | :-----------: |
| model1 | 1.9821 |  33.35%  |    53.39s     |

- Conv2D: 32, 3, 1, 'valid'.
- Flatten.
- Dense: 10, 'softmax'.

<br />

#### Loss and Accuracy Plots

<img src="plots/ex2/keras/model1_loss.png" height="240" />
<img src="plots/ex2/keras/model1_accuracy.png" height="240" />

<br />

Not only the accuracy is low, but both loss and accuracy plots show overfitting. This model is not satisfactory at all. 

<br />

#### Confusion Matrix

<img src="plots/ex2/keras/model1_confusion_matrix.png" height="400" />

<br />

The confusion matrix looks bad: there is no high value diagonal. It adds a proof that the model is bad.

<br />

#### 10 Worst Classified Images

| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                      Image                      |
| :---: | :--------: | :--------: | :-------: | :---------------------------------------------: |
|  10   |    7451    |    Car     |   Truck   | <img src="ten_worst/ex2/keras/model1/10.png" /> |
|   9   |    7196    |    Ship    |   Plane   | <img src="ten_worst/ex2/keras/model1/9.png" />  |
|   8   |    8309    |    Frog    |   Deer    | <img src="ten_worst/ex2/keras/model1/8.png" />  |
|   7   |    5918    |   Truck    |    Car    | <img src="ten_worst/ex2/keras/model1/7.png" />  |
|   6   |    7811    |    Ship    |   Plane   | <img src="ten_worst/ex2/keras/model1/6.png" />  |
|   5   |    1889    |    Ship    |   Truck   | <img src="ten_worst/ex2/keras/model1/5.png" />  |
|   4   |    7455    |   Truck    |    Car    | <img src="ten_worst/ex2/keras/model1/4.png" />  |
|   3   |    4971    |    Car     |   Truck   | <img src="ten_worst/ex2/keras/model1/3.png" />  |
|   2   |    3024    |    Ship    |   Plane   | <img src="ten_worst/ex2/keras/model1/2.png" />  |
|   1   |    5008    |   Horse    |   Deer    | <img src="ten_worst/ex2/keras/model1/1.png" />  |

<br />

### 1.3. Comparison

|    Model    | Accuracy |  Time  |
| :---------: | :------: | :----: |
| best lab3.3 |  44.50%  | 46.06s |
|   model1    |  33.35%  | 53.39s |

Our best lab3.3. model gives a more than 10% better accuracy than our first CNN model. Let's try to improve it.

<br />

### 1.4. Model Improvement

#### Model Summaries

Legend:
- Conv2D: num_filters, kernel_size, stride, padding.
- MaxPooling: pool_size, stride, padding.
- Dense: units, activation_function.

|   ID   | Architecture                                                                                                                                                                                                                                                                                                                   |  Loss  | Accuracy | Training time |
| :----: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: | :------: | :-----------: |
| model5 | - Conv2D: 64, 5, 1, 'valid' <br /> - Conv2D: 128, 5, 1, 'valid' <br /> - BatchNorm <br /> - MaxPooling: 2, 2, 'valid' <br /> - Conv2D: 256, 5, 1, 'valid' <br /> - Flatten <br /> - Dense: 10, 'softmax'                                                                                                                       | 1.5979 |  55.96%  |    200.59s    |
| model6 | - Conv2D: 64, 5, 1, 'valid' <br /> - MaxPooling: 2, 2, 'valid' <br /> - BatchNorm <br /> - Conv2D: 128, 5, 1, 'valid' <br /> - MaxPooling: 2, 2, 'valid' <br />  - BatchNorm <br /> - Flatten <br /> - Dense: 128, 'relu' <br /> - Dense: 10, 'softmax'                                                                        | 3.3476 |  63.78%  |    95.43s     |
| model7 | - Conv2D: 64, 5, 1, 'valid' <br /> - Conv2D: 64, 5, 1, 'valid' <br /> - MaxPooling: 2, 2, 'valid' <br /> - BatchNorm <br /> - Conv2D: 128, 5, 1, 'valid' <br /> - Conv2D: 128, 5, 1, 'valid' <br /> - MaxPooling: 2, 2, 'valid' <br />  - BatchNorm <br /> - Flatten <br /> - Dense: 128, 'relu' <br /> - Dense: 10, 'softmax' | 1.3447 |  68.57%  |    158.78s    |

The accuracy increases through the different models. What's more, the one giving the best accuracy (model7) is not even the slower one. 68% might not seem a great accuracy, but it doubled since model1 which is good news.


<br />

#### Loss, Accuracy Plots and Confusion Matrices

|   ID   |                         Loss Plot                          |                         Accuracy Plot                          |                            Confusion Matrix                            |
| :----: | :--------------------------------------------------------: | :------------------------------------------------------------: | :--------------------------------------------------------------------: |
| model5 | <img src="plots/ex2/keras/model5_loss.png" height="150" /> | <img src="plots/ex2/keras/model5_accuracy.png" height="150" /> | <img src="plots/ex2/keras/model5_confusion_matrix.png" height="150" /> |
| model6 | <img src="plots/ex2/keras/model6_loss.png" height="150" /> | <img src="plots/ex2/keras/model6_accuracy.png" height="150" /> | <img src="plots/ex2/keras/model6_confusion_matrix.png" height="150" /> |
| model7 | <img src="plots/ex2/keras/model7_loss.png" height="150" /> | <img src="plots/ex2/keras/model7_accuracy.png" height="150" /> | <img src="plots/ex2/keras/model7_confusion_matrix.png" height="150" /> |

Compared to model1, the confusion matrices are much better: they all present the famous diagonal.

Even though we managed to improve the accuracy, it's becoming harder to fight overfitting. Indeed, all 3 models show a lot of overfitting. (Notice that the last one looks like it's slightly less overfitting???)

<br />

#### 10 Worst Classified Images

|       |   model5   |            |           |                                                 |   model6   |            |           |                                                 |   model7   |            |           |                                                 |
| :---: | :--------: | :--------: | :-------: | :---------------------------------------------: | :--------: | :--------: | :-------: | :---------------------------------------------: | :--------: | :--------: | :-------: | :---------------------------------------------: |
| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                      Image                      | Image Idx. | Pred. Cat. | Act. Cat. |                      Image                      | Image Idx. | Pred. Cat. | Act. Cat. |                      Image                      |
|  10   |    9935    |   Truck    |    Car    | <img src="ten_worst/ex2/keras/model5/10.png" /> |    5122    |    Cat     |   Frog    | <img src="ten_worst/ex2/keras/model6/10.png" /> |    5251    |    Cat     |    Dog    | <img src="ten_worst/ex2/keras/model7/10.png" /> |
|   9   |    2284    |   Plane    |   Ship    | <img src="ten_worst/ex2/keras/model5/9.png" />  |    9206    |   Plane    |   Ship    | <img src="ten_worst/ex2/keras/model6/9.png" />  |    1090    |    Ship    |   Plane   | <img src="ten_worst/ex2/keras/model7/9.png" />  |
|   8   |    5040    |    Frog    |    Cat    | <img src="ten_worst/ex2/keras/model5/8.png" />  |    7236    |   Truck    |    Car    | <img src="ten_worst/ex2/keras/model6/8.png" />  |    8201    |    Frog    |    Cat    | <img src="ten_worst/ex2/keras/model7/8.png" />  |
|   7   |    9407    |    Car     |   Frog    | <img src="ten_worst/ex2/keras/model5/7.png" />  |    7329    |   Truck    |    Car    | <img src="ten_worst/ex2/keras/model6/7.png" />  |    6760    |    Cat     |    Dog    | <img src="ten_worst/ex2/keras/model7/7.png" />  |
|   6   |    5546    |    Deer    |   Horse   | <img src="ten_worst/ex2/keras/model5/6.png" />  |    3253    |    Ship    |   Truck   | <img src="ten_worst/ex2/keras/model6/6.png" />  |    6002    |   Plane    |   Bird    | <img src="ten_worst/ex2/keras/model7/6.png" />  |
|   5   |    1433    |    Deer    |   Horse   | <img src="ten_worst/ex2/keras/model5/5.png" />  |    2060    |    Cat     |    Dog    | <img src="ten_worst/ex2/keras/model6/5.png" />  |    783     |    Ship    |   Frog    | <img src="ten_worst/ex2/keras/model7/5.png" />  |
|   4   |    8915    |    Car     |   Truck   | <img src="ten_worst/ex2/keras/model5/4.png" />  |    5729    |    Car     |   Ship    | <img src="ten_worst/ex2/keras/model6/4.png" />  |    1631    |   Truck    |    Car    | <img src="ten_worst/ex2/keras/model7/4.png" />  |
|   3   |    757     |   Plane    |   Ship    | <img src="ten_worst/ex2/keras/model5/3.png" />  |    447     |    Ship    |   Plane   | <img src="ten_worst/ex2/keras/model6/3.png" />  |    4721    |   Plane    |    Car    | <img src="ten_worst/ex2/keras/model7/3.png" />  |
|   2   |    1272    |    Frog    |   Deer    | <img src="ten_worst/ex2/keras/model5/2.png" />  |    1650    |    Bird    |    Dog    | <img src="ten_worst/ex2/keras/model6/2.png" />  |    1653    |    Car     |   Truck   | <img src="ten_worst/ex2/keras/model7/2.png" />  |
|   1   |    3549    |   Truck    |    Car    | <img src="ten_worst/ex2/keras/model5/1.png" />  |    3402    |    Frog    |   Deer    | <img src="ten_worst/ex2/keras/model6/1.png" />  |    9148    |    Dog     |    Cat    | <img src="ten_worst/ex2/keras/model7/1.png" />  |

<br /><br />

## 3. Data Augmentation on CIFAR10 dataset

Our last model (model7) is the best one we could get with a CNN so far, but it shows overfitting. One way to reduce overfitting is to increase the size of the training dataset. Let's try to improve our model by doing data augmentation on our dataset.

### 3.1. Results

#### Model Summaries

All the models described below are heritated from model7, so we will only specify the parameters of the `ImageDataGenerator` in the Architecture column.

|         ID          | Architecture                                                                                                                             | Epochs |  Loss  | Accuracy |          Training time           |
| :-----------------: | :--------------------------------------------------------------------------------------------------------------------------------------- | :----: | :----: | :------: | :------------------------------: |
|       model8        | - horizontal_flip=True <br /> - height_shift_range=0.1 <br /> - width_shift_range=0.1 <br /> - rotation_range=10 <br /> - zoom_range=0.2 |   20   | 0.9139 |  69.78%  |             340.25s              |
|       model9        | - horizontal_flip=True <br /> - height_shift_range=0.1 <br /> - width_shift_range=0.1                                                    |   20   | 0.8267 |  72.08%  |             323.14s              |
| model10 (model9bis) | - horizontal_flip=True <br /> - height_shift_range=0.1 <br /> - width_shift_range=0.1                                                    |  100   | 0.7761 |  76.33%  | 1557.88s (311.58s for 20 epochs) |

<br />

|        ID        |                                 Images                                  |
| :--------------: | :---------------------------------------------------------------------: |
|     original     |        <img src="plots/ex3/keras/dataset_og.png" height="100" />        |
|      model8      | <img src="plots/ex3/keras/model8_dataset_augmented.png" height="100" /> |
| model9 & model10 | <img src="plots/ex3/keras/model9_dataset_augmented.png" height="100" /> |

<br />

For model8, the parameters were chosen according to what is plausible for the subjects of our images. That's why we, for example, did not add a vertical flip (an upside down ship or truck makes no sense).
As for model9(bis), some parameters were removed as a test.
Finally, model10 is actually model9, but trained on 100 epochs.

When it comes to perfomance, the accuracy gets better as the training time goes down, which suggests that each model might be better than the previous one.
To confirm it, let's take a look at the plots.

<br />

#### Loss, Accuracy Plots and Confusion Matrices

|   ID    |                          Loss Plot                          |                          Accuracy Plot                          |                            Confusion Matrix                             |
| :-----: | :---------------------------------------------------------: | :-------------------------------------------------------------: | :---------------------------------------------------------------------: |
| model8  | <img src="plots/ex3/keras/model8_loss.png" height="150" />  | <img src="plots/ex3/keras/model8_accuracy.png" height="150" />  | <img src="plots/ex3/keras/model8_confusion_matrix.png" height="150" />  |
| model9  | <img src="plots/ex3/keras/model9_loss.png" height="150" />  | <img src="plots/ex3/keras/model9_accuracy.png" height="150" />  | <img src="plots/ex3/keras/model9_confusion_matrix.png" height="150" />  |
| model10 | <img src="plots/ex3/keras/model10_loss.png" height="150" /> | <img src="plots/ex3/keras/model10_accuracy.png" height="150" /> | <img src="plots/ex3/keras/model10_confusion_matrix.png" height="150" /> |

<br />

It was observed in the previous section (Model summaries) that our new models were providing better accuracies, but our main problem that we are trying to solve with data augmentation is overfitting. 
Looking at the model8 accuracy plot, we can see that the validation curve is above the test one. It could mean that the model is underfitting but the gap is not that big.

With model9, we managed to slightly reduce the gap a bit. Since the curves look like they keep going up, we trained it again on more epochs (model10).
Even though the test accuracy oscillates around the train accuracy, the test accuracy follows closely the train accuracy which means less/no overfitting.

<br />

#### 10 Worst Classified Images

|       |   model8   |            |           |            |                                                 |   model9   |            |           |            |                                                 |  model10   |            |           |            |                                                  |
| :---: | :--------: | :--------: | :-------: | :--------: | :---------------------------------------------: | :--------: | :--------: | :-------: | :--------: | :---------------------------------------------: | :--------: | :--------: | :-------: | :--------: | :----------------------------------------------: |
| Rank  | Image Idx. | Pred. Cat. | Act. Cat. | Prob. Act. |                      Image                      | Image Idx. | Pred. Cat. | Act. Cat. | Prob. Act. |                      Image                      | Image Idx. | Pred. Cat. | Act. Cat. | Prob. Act. |                      Image                       |
|  10   |    1396    |   Truck    |    Car    |   0.4971   | <img src="ten_worst/ex3/keras/model8/10.png" /> |    7279    |    Cat     |    Dog    |   0.4975   | <img src="ten_worst/ex3/keras/model9/10.png" /> |    6832    |   Truck    |    Car    |   0.4924   | <img src="ten_worst/ex3/keras/model10/10.png" /> |
|   9   |    2063    |   Truck    |    Car    |   0.4951   | <img src="ten_worst/ex3/keras/model8/9.png" />  |    6267    |    Car     |   Truck   |   0.4974   | <img src="ten_worst/ex3/keras/model9/9.png" />  |    1051    |    Frog    |   Bird    |   0.4839   | <img src="ten_worst/ex3/keras/model10/9.png" />  |
|   8   |    3404    |    Cat     |    Dog    |   0.4934   | <img src="ten_worst/ex3/keras/model8/8.png" />  |    4005    |    Cat     |    Dog    |   0.4965   | <img src="ten_worst/ex3/keras/model9/8.png" />  |    3094    |   Plane    |   Bird    |   0.4833   | <img src="ten_worst/ex3/keras/model10/8.png" />  |
|   7   |    9141    |    Frog    |    Cat    |   0.4921   | <img src="ten_worst/ex3/keras/model8/7.png" />  |    2964    |    Ship    |   Plane   |   0.4941   | <img src="ten_worst/ex3/keras/model9/7.png" />  |    8884    |   Horse    |   Plane   |   0.4832   | <img src="ten_worst/ex3/keras/model10/7.png" />  |
|   6   |    8595    |   Truck    |    Car    |   0.4904   | <img src="ten_worst/ex3/keras/model8/6.png" />  |    8437    |    Deer    |   Horse   |   0.4930   | <img src="ten_worst/ex3/keras/model9/6.png" />  |    2153    |   Plane    |   Ship    |   0.4783   | <img src="ten_worst/ex3/keras/model10/6.png" />  |
|   5   |    9102    |    Deer    |   Horse   |   0.4885   | <img src="ten_worst/ex3/keras/model8/5.png" />  |    9295    |    Dog     |    Cat    |   0.4929   | <img src="ten_worst/ex3/keras/model9/5.png" />  |    5805    |   Truck    |    Car    |   0.4771   | <img src="ten_worst/ex3/keras/model10/5.png" />  |
|   4   |    7384    |    Dog     |    Cat    |   0.4873   | <img src="ten_worst/ex3/keras/model8/4.png" />  |    9649    |    Cat     |    Dog    |   0.4925   | <img src="ten_worst/ex3/keras/model9/4.png" />  |    5240    |   Truck    |    Car    |   0.4769   | <img src="ten_worst/ex3/keras/model10/4.png" />  |
|   3   |    6055    |    Ship    |   Plane   |   0.4821   | <img src="ten_worst/ex3/keras/model8/3.png" />  |    6228    |   Horse    |   Deer    |   0.4904   | <img src="ten_worst/ex3/keras/model9/3.png" />  |    7942    |    Car     |   Truck   |   0.4742   | <img src="ten_worst/ex3/keras/model10/3.png" />  |
|   2   |    6399    |   Truck    |    Car    |   0.4803   | <img src="ten_worst/ex3/keras/model8/2.png" />  |    1733    |    Dog     |    Cat    |   0.4874   | <img src="ten_worst/ex3/keras/model9/2.png" />  |    5085    |    Frog    |   Bird    |   0.4732   | <img src="ten_worst/ex3/keras/model10/2.png" />  |
|   1   |    4125    |   Plane    |   Ship    |   0.4783   | <img src="ten_worst/ex3/keras/model8/1.png" />  |    3130    |    Dog     |    Cat    |   0.4869   | <img src="ten_worst/ex3/keras/model9/1.png" />  |    4421    |    Dog     |   Bird    |   0.4692   | <img src="ten_worst/ex3/keras/model10/1.png" />  |

By analysing each ranking, we can observe that
- model8 tends to mistake Car as Truck (4 times)
- model9/10 also (Car as Truck 3 times, Truck as Car 1 time), and Cat as Dog, Dog as Cat (3 times each)

These classes are respectively of the same type of subject, and kind of look alike. Therefore, they are not aberrant mistakes.

<br /><br />

### 3.2. Comparison

To conclude this part, let's compare our new best model (model10) with the ones of our original model (model7) that was the base.

|   ID    |                          Accuracy Plot                          | Accuracy |          Training time           |
| :-----: | :-------------------------------------------------------------: | :------: | :------------------------------: |
| model7  | <img src="plots/ex2/keras/model7_accuracy.png" height="150" />  |  68.57%  |             158.78s              |
| model10 | <img src="plots/ex3/keras/model10_accuracy.png" height="150" /> |  76.33%  | 1557.88s (311.58s for 20 epochs) |

In spite of an almost twice longer training time, model10 overperforms model7. 
Indeed, model10 provides an accuracy that is almost 8% higher and the problem of overfitting seems solved
(even if the plots are not on the same number of epochs, we do not need more epochs to see that the test accuracy will not get higher in model7).


<br /><br />

## 4. Transfer learning / Fine-tuning on CIFAR10 dataset

### 4.1. Results

For this part, we are going to use ResNet50 pre-trained on ImageNet. 
We want to specify our input shape and remove the classifier to add our own so the model can classify 10 classes.

#### Model Summaries

Now that we know data augmentation helps improve results, we want to try fine-tuning with and without data augmentation.

|     ID     |   Data Augmentation   |  Loss  | Accuracy | Training time |
| :--------: | :-------------------: | :----: | :------: | :-----------: |
|  MyResNet  |          No           | 3.0612 |  73.63%  |   1153.49s    |
| MyResNetDA | Yes, same than model9 | 2.4440 |  76.15%  |   1172.44s    |

<br />

We expected MyResNetDA to provide a higher accuracy and it did, but very slightly (+3%).
As ResNet50 has 50 layers and none of them were frozen, it is trained entirely and so the training time is quite long (about 20 minutes).

<br />

#### Loss, Accuracy Plots and Confusion Matrices

|     ID     |                           Loss Plot                            |                           Accuracy Plot                            |                              Confusion Matrix                              |
| :--------: | :------------------------------------------------------------: | :----------------------------------------------------------------: | :------------------------------------------------------------------------: |
|  MyResNet  |  <img src="plots/ex4/keras/MyResNet/loss.png" height="150" />  |  <img src="plots/ex4/keras/MyResNet/accuracy.png" height="150" />  |  <img src="plots/ex4/keras/MyResNet/confusion_matrix.png" height="150" />  |
| MyResNetDA | <img src="plots/ex4/keras/MyResNetDA/loss.png" height="150" /> | <img src="plots/ex4/keras/MyResNetDA/accuracy.png" height="150" /> | <img src="plots/ex4/keras/MyResNetDA/confusion_matrix.png" height="150" /> |

<br />

MyResNet's accuracy plot shows overfitting, whereas data augmentation seems to do its job in MyResNetDA.
Indeed, the test accuracy seems very close to train accuracy until around the 9th epoch (we couldn't figure out the drop). Then it gets back on track towards the end.

Although both confusion matrices look OK, with the diagonal representing the true positives, we cannot say the same about loss plots.
They look abnormal, and we unfortunately could not figure out why.

After many tries, with different types of normalization and/or layer freezing, we did not manage to get interesting/better results.

<br />

#### 10 Worst Classified Images

|       |  MyResNet  |            |           |            |                                                   | MyResNetDA |            |           |            |                                                     |
| :---: | :--------: | :--------: | :-------: | :--------: | :-----------------------------------------------: | :--------: | :--------: | :-------: | :--------: | :-------------------------------------------------: |
| Rank  | Image Idx. | Pred. Cat. | Act. Cat. | Prob. Act. |                       Image                       | Image Idx. | Pred. Cat. | Act. Cat. | Prob. Act. |                        Image                        |
|  10   |    3029    |   Truck    |    Car    |   0.4975   | <img src="ten_worst/ex4/keras/MyResNet/10.png" /> |    1771    |    Deer    |   Bird    |   0.4934   | <img src="ten_worst/ex4/keras/MyResNetDA/10.png" /> |
|   9   |    7970    |    Dog     |    Cat    |   0.4925   | <img src="ten_worst/ex4/keras/MyResNet/9.png" />  |    7639    |    Car     |   Truck   |   0.4913   | <img src="ten_worst/ex4/keras/MyResNetDA/9.png" />  |
|   8   |    9778    |    Dog     |    Cat    |   0.4895   | <img src="ten_worst/ex4/keras/MyResNet/8.png" />  |    6581    |   Truck    |    Car    |   0.4890   | <img src="ten_worst/ex4/keras/MyResNetDA/8.png" />  |
|   7   |    5432    |    Ship    |   Truck   |   0.4883   | <img src="ten_worst/ex4/keras/MyResNet/7.png" />  |    7093    |    Deer    |   Bird    |   0.4852   | <img src="ten_worst/ex4/keras/MyResNetDA/7.png" />  |
|   6   |    778     |    Ship    |   Plane   |   0.4871   | <img src="ten_worst/ex4/keras/MyResNet/6.png" />  |    8761    |    Ship    |   Plane   |   0.4848   | <img src="ten_worst/ex4/keras/MyResNetDA/6.png" />  |
|   5   |    169     |    Ship    |   Plane   |   0.4856   | <img src="ten_worst/ex4/keras/MyResNet/5.png" />  |    1549    |   Truck    |    Car    |   0.4822   | <img src="ten_worst/ex4/keras/MyResNetDA/5.png" />  |
|   4   |    2744    |    Car     |   Truck   |   0.4838   | <img src="ten_worst/ex4/keras/MyResNet/4.png" />  |    4223    |   Plane    |   Ship    |   0.4812   | <img src="ten_worst/ex4/keras/MyResNetDA/4.png" />  |
|   3   |     55     |   Plane    |   Ship    |   0.4825   | <img src="ten_worst/ex4/keras/MyResNet/3.png" />  |    2809    |    Frog    |   Deer    |   0.4778   | <img src="ten_worst/ex4/keras/MyResNetDA/3.png" />  |
|   2   |    7605    |    Deer    |   Bird    |   0.4733   | <img src="ten_worst/ex4/keras/MyResNet/2.png" />  |    3182    |   Plane    |   Ship    |   0.4763   | <img src="ten_worst/ex4/keras/MyResNetDA/2.png" />  |
|   1   |    7714    |    Dog     |    Cat    |   0.4722   | <img src="ten_worst/ex4/keras/MyResNet/1.png" />  |    2879    |    Ship    |   Plane   |   0.4752   | <img src="ten_worst/ex4/keras/MyResNetDA/1.png" />  |

By analysing each ranking, we can observe that
- MyResNet tends to mistake Cat as Dog (3 times)
- MyResNetDA tends to mistake Ship as Truck and Truck as Ship (3 times in total)

In the same way than in the previous part about data augmentation, the mistakes done by our models are "understanble" since it confuses subjects that are of the "same type".

<br /><br />

### 4.1. Comparison

To conclude this part, let's compare the model improvements (model10 and MyResNetDA), and our model from part 2 (model7).

|    ID    |                          Accuracy Plot                           | Accuracy |          Training time           |
| :------: | :--------------------------------------------------------------: | :------: | :------------------------------: |
|  model7  |  <img src="plots/ex2/keras/model7_accuracy.png" height="150" />  |  68.57%  |             158.78s              |
| model10  | <img src="plots/ex3/keras/model10_accuracy.png" height="150" />  |  76.33%  | 1557.88s (311.58s for 20 epochs) |
| MyResNet | <img src="plots/ex4/keras/MyResNet/accuracy.png" height="150" /> |  76.15%  |             1172.44s             |

<br />

Our improved models (model10 and MyResNetDA) provides about the same accuracy but MyResNetDA overfits, unlike model10.
What is more, MyResNetDA is almost 4 times slower.

As a conclusion, MyResNetDA was able to improve model7 but could not compete with model10.

<br /><br />

# PyTorch

## 1. Conv2D Neural Network on MNIST Dataset

### 1.1. First CNN

#### Model Summary

|   ID   |  Loss  | Accuracy | Training Time |
| :----: | :----: | :------: | :-----------: |
| model1 | 0.2996 |  91.74%  |    92.76s     |

- Conv2D: 32, 3, 1, 'valid'.
- Flatten.
- Dense: 10 ('softmax').

*N.B.:* This is the same model1 as with Keras to have the same base, but it won't be improved the same way as Keras, since we chose to improve it the most relevant way possible. This will allow us to have other CNN architectures giving good accuracies.

#### Loss and Accuracy Plots

<img src="plots/ex1/pytorch/model1_loss.png" height="240" />
<img src="plots/ex1/pytorch/model1_accuracy.png" height="240" />

These plots may show some overfitting, but not much.

#### Confusion Matrix

<img src="plots/ex1/pytorch/model1_confusion_matrix.png" height="400" />

As in Keras, the confusion matrix shows that most the images are well classified (the diagonal). The most misclassified digits are:
- 7 as 9 (35)
- 4 as 9 (38)
- 5 as 3 (51)

#### Nota Bene

In PyTorch, the Softmax activation is already done by the CrossEntropyLoss criterion, as mentionned in the [official documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html): "Note that this is equivalent to the combination of LogSoftmax and NLLLoss." (That's why we put softmax between parenthesis.)

As we didn't know this at first, we did a first version of this model with a softmax activation on the linear layer. The results were drastically different.

<img src="plots/ex1/pytorch/old_model1_loss.png" height="240" />
<img src="plots/ex1/pytorch/old_model1_accuracy.png" height="240" />

<img src="plots/ex1/pytorch/old_model1_confusion_matrix.png" height="400" />

The confusion matrix shows that not only the elements are not well classified, but also some classes are nerver predicted.

#### 10 Worst Classified Images

The same way as before, we're going to determine the 10 worst classified images by the model.

| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                       Image                       |
| :---: | :--------: | :--------: | :-------: | :-----------------------------------------------: |
|  10   |    6885    |     6      |     2     | <img src="ten_worst/ex1/pytorch/model1/10.png" /> |
|   9   |    6599    |     1      |     7     | <img src="ten_worst/ex1/pytorch/model1/9.png" />  |
|   8   |    9487    |     6      |     2     | <img src="ten_worst/ex1/pytorch/model1/8.png" />  |
|   7   |    3189    |     4      |     7     | <img src="ten_worst/ex1/pytorch/model1/7.png" />  |
|   6   |    5688    |     9      |     7     | <img src="ten_worst/ex1/pytorch/model1/6.png" />  |
|   5   |    1940    |     0      |     5     | <img src="ten_worst/ex1/pytorch/model1/5.png" />  |
|   4   |    1017    |     2      |     6     | <img src="ten_worst/ex1/pytorch/model1/4.png" />  |
|   3   |    1310    |     7      |     3     | <img src="ten_worst/ex1/pytorch/model1/3.png" />  |
|   2   |    3682    |     6      |     2     | <img src="ten_worst/ex1/pytorch/model1/2.png" />  |
|   1   |    9916    |     9      |     7     | <img src="ten_worst/ex1/pytorch/model1/1.png" />  |

<br />

### 1.2. Model Improvement

### 1.2.1. A New Architecture

Once again, we are going to complexify our architecture.

#### Model Summary

|   ID   |  Loss  | Accuracy | Training Time |
| :----: | :----: | :------: | :-----------: |
| model2 | 0.1276 |  97.20%  |    97.12s     |

- `Conv2D: 64, 3, 1, 'valid'.`
- Conv2D: 32, 3, 1, 'valid'.
- `MaxPooling: 2, 1, 'valid'.`
- `Conv2D: 16, 3, 1, 'valid'.`
- Flatten.
- Dense: 10 ('softmax').

This model is definitely better than the first one. For only a few seconds longer, it gives a 6 to 7 percent better accuracy.

#### Loss and Accuracy Plots

<img src="plots/ex1/pytorch/model2_loss.png" height="240" />
<img src="plots/ex1/pytorch/model2_accuracy.png" height="240" />

However this time, there is an obvious overfitting detected as the training loss keeps descending but the validation one is ascending.

#### Confusion Matrix

<img src="plots/ex1/pytorch/model2_confusion_matrix.png" height="400" />

However, the confusion matrix—and also the accuracy obviously—still shows that most of the images are classified correctly.

#### 10 Worst Classified Images

| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                       Image                       |
| :---: | :--------: | :--------: | :-------: | :-----------------------------------------------: |
|  10   |    5593    |     6      |     0     | <img src="ten_worst/ex1/pytorch/model2/10.png" /> |
|   9   |    5176    |     4      |     8     | <img src="ten_worst/ex1/pytorch/model2/9.png" />  |
|   8   |     8      |     6      |     5     | <img src="ten_worst/ex1/pytorch/model2/8.png" />  |
|   7   |    2370    |     6      |     0     | <img src="ten_worst/ex1/pytorch/model2/7.png" />  |
|   6   |    6532    |     5      |     0     | <img src="ten_worst/ex1/pytorch/model2/6.png" />  |
|   5   |    9614    |     5      |     3     | <img src="ten_worst/ex1/pytorch/model2/5.png" />  |
|   4   |    8069    |     1      |     2     | <img src="ten_worst/ex1/pytorch/model2/4.png" />  |
|   3   |    6847    |     4      |     6     | <img src="ten_worst/ex1/pytorch/model2/3.png" />  |
|   2   |    5228    |     4      |     6     | <img src="ten_worst/ex1/pytorch/model2/2.png" />  |
|   1   |    965     |     0      |     6     | <img src="ten_worst/ex1/pytorch/model2/1.png" />  |

With this model, we're starting to understand why the neural network is wrong sometimes, as one might be wrong the same way on some images.

<br />

### 1.2.2. Fighting Against Overfitting

This time, let's build a model with data normalization, to prevent overfitting. And then, try and improve its accuracy.

#### Models Summaries

|   ID   | Architecture                                                                                                                                                                                                                                                             |  Loss  | Accuracy | Training time |
| :----: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: | :------: | :-----------: |
| model3 | - Conv2D: 64, 3, 1, 'valid'. <br /> - Conv2D: 32, 3, 1, 'valid'. <br /> - `Dropout.` <br /> - `Activation: 'relu'.` <br /> - MaxPooling: 2, 1, 'valid'. <br /> - Conv2D: 16, 3, 1, 'valid'. <br /> - Flatten. <br /> - Dense: 10 ('softmax').                            | 0.0456 |  98.55%  |    98.83s     |
| model4 | - Conv2D: 64, 3, 1, 'valid'. <br /> - Conv2D: 32, 3, 1, 'valid'. <br /> - Dropout. <br /> - Activation: 'relu'. <br /> - MaxPooling: 2, 1, 'valid'. <br /> - Conv2D: 16, 3, 1, 'valid'. <br /> - Flatten. <br /> - `Dense: 128, 'relu'.` <br /> - Dense: 10 ('softmax'). | 0.0432 |  98.76%  |    103.71s    |

The 2 new models provide rather identical results, slightly better than the model2 ones.

|   ID   |                          Loss Plot                           |                          Accuracy Plot                           |                             Confusion Matrix                             |
| :----: | :----------------------------------------------------------: | :--------------------------------------------------------------: | :----------------------------------------------------------------------: |
| model3 | <img src="plots/ex1/pytorch/model3_loss.png" height="150" /> | <img src="plots/ex1/pytorch/model3_accuracy.png" height="150" /> | <img src="plots/ex1/pytorch/model3_confusion_matrix.png" height="150" /> |
| model4 | <img src="plots/ex1/pytorch/model4_loss.png" height="150" /> | <img src="plots/ex1/pytorch/model4_accuracy.png" height="150" /> | <img src="plots/ex1/pytorch/model4_confusion_matrix.png" height="150" /> |

Here, model3 seems to have less overfitting, but it depends also on the run as the difference is very subtle.

|       |   model3   |            |           |                                                   |   model4   |            |           |                                                   |
| :---: | :--------: | :--------: | :-------: | :-----------------------------------------------: | :--------: | :--------: | :-------: | :-----------------------------------------------: |
| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                       Image                       | Image Idx. | Pred. Cat. | Act. Cat. |                       Image                       |
|  10   |    1147    |     7      |     4     | <img src="ten_worst/ex1/pytorch/model3/10.png" /> |    2953    |     5      |     3     | <img src="ten_worst/ex1/pytorch/model4/10.png" /> |
|   9   |    8059    |     1      |     2     | <img src="ten_worst/ex1/pytorch/model3/9.png" />  |    5937    |     3      |     5     | <img src="ten_worst/ex1/pytorch/model4/9.png" />  |
|   8   |    9664    |     7      |     2     | <img src="ten_worst/ex1/pytorch/model3/8.png" />  |    6091    |     5      |     9     | <img src="ten_worst/ex1/pytorch/model4/8.png" />  |
|   7   |    4063    |     5      |     6     | <img src="ten_worst/ex1/pytorch/model3/7.png" />  |    359     |     4      |     9     | <img src="ten_worst/ex1/pytorch/model4/7.png" />  |
|   6   |    4838    |     5      |     6     | <img src="ten_worst/ex1/pytorch/model3/6.png" />  |    2130    |     9      |     4     | <img src="ten_worst/ex1/pytorch/model4/6.png" />  |
|   5   |    3073    |     2      |     1     | <img src="ten_worst/ex1/pytorch/model3/5.png" />  |    3030    |     0      |     6     | <img src="ten_worst/ex1/pytorch/model4/5.png" />  |
|   4   |    2035    |     3      |     5     | <img src="ten_worst/ex1/pytorch/model3/4.png" />  |    4860    |     9      |     4     | <img src="ten_worst/ex1/pytorch/model4/4.png" />  |
|   3   |    217     |     5      |     6     | <img src="ten_worst/ex1/pytorch/model3/3.png" />  |    8277    |     8      |     3     | <img src="ten_worst/ex1/pytorch/model4/3.png" />  |
|   2   |    9755    |     5      |     8     | <img src="ten_worst/ex1/pytorch/model3/2.png" />  |    8521    |     1      |     2     | <img src="ten_worst/ex1/pytorch/model4/2.png" />  |
|   1   |    9015    |     2      |     7     | <img src="ten_worst/ex1/pytorch/model3/1.png" />  |    6081    |     5      |     9     | <img src="ten_worst/ex1/pytorch/model4/1.png" />  |

<br /><br />

## 2. Conv2D Neural Network on CIFAR10 Dataset

### 2.1. First CNN

#### Model Summary

For the first run, we chose once again to try the model1 on the new dataset

|   ID   |  Loss  | Accuracy | Training Time |
| :----: | :----: | :------: | :-----------: |
| model1 | 2.1902 |  30.53%  |    100.20s    |

- Conv2D: 32, 3, 1, 'valid'.
- Flatten.
- Dense: 10 ('softmax').

For the first time, we obtain bad results from a model. The color images were much more complicated to analyse.

#### Loss and Accuracy Plots

<img src="plots/ex2/pytorch/model1_loss.png" height="240" />
<img src="plots/ex2/pytorch/model1_accuracy.png" height="240" />

Also, the overfitting is already tremendous.

#### Confusion Matrix

<img src="plots/ex2/pytorch/model1_confusion_matrix.png" height="400" />

To confirm everything we've seen so far, the confusion matrix shows that the predictions are very far from the actual values. We can barely distinguish the diagonal.

#### 10 Worst Classified Images

| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                       Image                       |
| :---: | :--------: | :--------: | :-------: | :-----------------------------------------------: |
|  10   |    3650    |    Ship    |   Plane   | <img src="ten_worst/ex2/pytorch/model1/10.png" /> |
|   9   |    9590    |    Ship    |   Plane   | <img src="ten_worst/ex2/pytorch/model1/9.png" />  |
|   8   |    2968    |    Ship    |   Plane   | <img src="ten_worst/ex2/pytorch/model1/8.png" />  |
|   7   |    9766    |    Ship    |   Plane   | <img src="ten_worst/ex2/pytorch/model1/7.png" />  |
|   6   |    7454    |    Ship    |   Plane   | <img src="ten_worst/ex2/pytorch/model1/6.png" />  |
|   5   |    3278    |    Ship    |   Plane   | <img src="ten_worst/ex2/pytorch/model1/5.png" />  |
|   4   |    4981    |   Plane    |    Car    | <img src="ten_worst/ex2/pytorch/model1/4.png" />  |
|   3   |    1651    |    Ship    |   Plane   | <img src="ten_worst/ex2/pytorch/model1/3.png" />  |
|   2   |    7431    |    Car     |   Truck   | <img src="ten_worst/ex2/pytorch/model1/2.png" />  |
|   1   |    2200    |    Car     |   Plane   | <img src="ten_worst/ex2/pytorch/model1/1.png" />  |

<br />

### 2.2. Model Improvement

### 2.2.1. A New Architecture

#### Model Summary

This time, the model4, one of the best models we've tested, is used.

|   ID   |  Loss  | Accuracy | Training Time |
| :----: | :----: | :------: | :-----------: |
| model4 | 1.2091 |  67.51%  |    107.47s    |

- Conv2D: 64, 3, 1, 'valid'.
- Conv2D: 32, 3, 1, 'valid'.
- Dropout.
- Activation: 'relu'.
- MaxPooling: 2, 1, 'valid'.
- Conv2D: 16, 3, 1, 'valid'.
- Flatten.
- Dense: 128, 'relu'.
- Dense: 10 ('softmax').

We might be lucky, but this model actualy improves drastically the results we've had with model1. The accuracy is already almost as good as the fourth model tested with Keras.

#### Loss and Accuracy Plots

<img src="plots/ex2/pytorch/model4_loss.png" height="240" />
<img src="plots/ex2/pytorch/model4_accuracy.png" height="240" />

On the other hand, the overfitting is still very present, though we might have limited it.

#### Confusion Matrix

<img src="plots/ex2/pytorch/model4_confusion_matrix.png" height="400" />

The confusion matrix also shows an important improvement as the diagonal is much more visible now.

#### 10 Worst Classified Images

| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                       Image                       |
| :---: | :--------: | :--------: | :-------: | :-----------------------------------------------: |
|  10   |    1692    |    Car     |   Truck   | <img src="ten_worst/ex2/pytorch/model4/10.png" /> |
|   9   |    1732    |    Car     |   Truck   | <img src="ten_worst/ex2/pytorch/model4/9.png" />  |
|   8   |    9854    |    Car     |   Truck   | <img src="ten_worst/ex2/pytorch/model4/8.png" />  |
|   7   |    4866    |    Car     |   Truck   | <img src="ten_worst/ex2/pytorch/model4/7.png" />  |
|   6   |    1829    |   Truck    |    Car    | <img src="ten_worst/ex2/pytorch/model4/6.png" />  |
|   5   |    3150    |   Truck    |    Car    | <img src="ten_worst/ex2/pytorch/model4/5.png" />  |
|   4   |    5041    |    Car     |   Truck   | <img src="ten_worst/ex2/pytorch/model4/4.png" />  |
|   3   |    6615    |    Dog     |   Horse   | <img src="ten_worst/ex2/pytorch/model4/3.png" />  |
|   2   |    6968    |   Plane    |   Ship    | <img src="ten_worst/ex2/pytorch/model4/2.png" />  |
|   1   |    3812    |    Car     |   Truck   | <img src="ten_worst/ex2/pytorch/model4/1.png" />  |

<br />

### 2.2.2. Going Further

#### Model Summary

Let's imagine a new model, inspired by model4, but trying to improve the final accuracy.

|   ID   |  Loss  | Accuracy | Training Time |
| :----: | :----: | :------: | :-----------: |
| model5 | 0.9670 |  70.11%  |    134.46s    |

- Conv2D: 64, 3, 1, 'valid'.
- Conv2D: 32, 3, 1, 'valid'.
- Dropout.
- Activation: 'relu'.
- MaxPooling: 2, 1, 'valid'.
- `Conv2D: 64, 3, 1, 'valid'.`
- `Conv2D: 32, 3, 1, 'valid'.`
- `Dropout.`
- `Activation: 'relu'.`
- `MaxPooling: 2, 1, 'valid'.`
- Conv2D: 16, 3, 1, 'valid'.
- Flatten.
- Dense: 128, 'relu'.
- `Dense: 256, 'relu'.`
- Dense: 10 ('softmax').

This model gives the best results for this dataset. On the other hand, the training time is starting to grow bigger.

#### Loss and Accuracy Plots

<img src="plots/ex2/pytorch/model5_loss.png" height="240" />
<img src="plots/ex2/pytorch/model5_accuracy.png" height="240" />

There is persistent overfitting but at this point, there is not much solution but to use data augmentation. We will try to focus on that next.

#### Confusion Matrix

<img src="plots/ex2/pytorch/model5_confusion_matrix.png" height="400" />

The confusion matrix is not perfect but it's the best we've had on this dataset, confirming that the model is better.

#### 10 Worst Classified Images

| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                       Image                       |
| :---: | :--------: | :--------: | :-------: | :-----------------------------------------------: |
|  10   |    5918    |   Truck    |    Car    | <img src="ten_worst/ex2/pytorch/model5/10.png" /> |
|   9   |    2322    |    Bird    |   Plane   | <img src="ten_worst/ex2/pytorch/model5/9.png" />  |
|   8   |    4766    |   Truck    |    Car    | <img src="ten_worst/ex2/pytorch/model5/8.png" />  |
|   7   |    3151    |    Car     |   Ship    | <img src="ten_worst/ex2/pytorch/model5/7.png" />  |
|   6   |     81     |   Truck    |    Car    | <img src="ten_worst/ex2/pytorch/model5/6.png" />  |
|   5   |    6342    |   Plane    |   Ship    | <img src="ten_worst/ex2/pytorch/model5/5.png" />  |
|   4   |    5416    |    Car     |   Truck   | <img src="ten_worst/ex2/pytorch/model5/4.png" />  |
|   3   |    4056    |    Ship    |   Plane   | <img src="ten_worst/ex2/pytorch/model5/3.png" />  |
|   2   |    5392    |    Car     |   Plane   | <img src="ten_worst/ex2/pytorch/model5/2.png" />  |
|   1   |    9981    |   Horse    |   Deer    | <img src="ten_worst/ex2/pytorch/model5/1.png" />  |

## 3. Data Augmentation

### Model Without Data Augmentation

We are now going to use a stable model based on `model5` that we previously saw, and matching the one we used on the Keras version. We will declare it as `PraisyNet` for no reason.

#### Model Summary

|    ID     | Data Augmentation |  Loss  | Accuracy | Training Time |
| :-------: | :---------------: | :----: | :------: | :-----------: |
| PraisyNet |        No         | 1.1851 |  69.80%  |    160.63s    |

The model architechture is the following:

- Layer1
    - Conv2d(NUM_CHANNELS, 64, 5, stride=1, padding=0)
    - Conv2d(64, 64, 5, stride=1, padding=0)
    - MaxPool2d(2, stride=2, padding=0)
    - LazyBatchNorm2d()
- Layer2
    - Conv2d(64, 128, 5, stride=1, padding=0)
    - Conv2d(128, 128, 5, stride=1, padding=0)
    - MaxPool2d(2, stride=2, padding=0)
    - LazyBatchNorm2d()
- Flatten
- Classifier
    - LazyLinear(128)
    - ReLU()
    - LazyLinear(NUM_CLASSES)

#### Loss and Accuracy Plots

<img src="plots/ex3/pytorch/praisynet_no_augmentation_loss.png" height="240" />
<img src="plots/ex3/pytorch/praisynet_no_augmentation_accuracy.png" height="240" />

As we have seen before with `model5`, there is a lot of overfitting here.

#### Confusion Matrix

<img src="plots/ex3/pytorch/praisynet_no_augmentation_confusion_matrix.png" height="400" />

The confusion matrix is good, confirming this is a good base model.

#### 10 Worst Classified Images

| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                                Image                                 |
| :---: | :--------: | :--------: | :-------: | :------------------------------------------------------------------: |
|  10   |    8981    |    Deer    |   Horse   | <img src="ten_worst/ex3/pytorch/praisynet_no_augmentation/10.png" /> |
|   9   |    9084    |    Cat     |    Dog    | <img src="ten_worst/ex3/pytorch/praisynet_no_augmentation/9.png" />  |
|   8   |    6814    |    Car     |   Truck   | <img src="ten_worst/ex3/pytorch/praisynet_no_augmentation/8.png" />  |
|   7   |    9461    |    Frog    |   Deer    | <img src="ten_worst/ex3/pytorch/praisynet_no_augmentation/7.png" />  |
|   6   |    7929    |    Bird    |   Plane   | <img src="ten_worst/ex3/pytorch/praisynet_no_augmentation/6.png" />  |
|   5   |    4032    |    Dog     |    Cat    | <img src="ten_worst/ex3/pytorch/praisynet_no_augmentation/5.png" />  |
|   4   |    2641    |    Frog    |   Deer    | <img src="ten_worst/ex3/pytorch/praisynet_no_augmentation/4.png" />  |
|   3   |    8187    |   Horse    |    Dog    | <img src="ten_worst/ex3/pytorch/praisynet_no_augmentation/3.png" />  |
|   2   |    1631    |   Truck    |    Car    | <img src="ten_worst/ex3/pytorch/praisynet_no_augmentation/2.png" />  |
|   1   |    8808    |    Cat     |   Frog    | <img src="ten_worst/ex3/pytorch/praisynet_no_augmentation/1.png" />  |

### Model With Data Augmentation

We are now going to use a stable model based on `model5` that we previously saw, and matching the one we used on the Keras version.

#### Model Summary

|    ID     | Data Augmentation |  Loss  | Accuracy |          Training Time          |
| :-------: | :---------------: | :----: | :------: | :-----------------------------: |
| PraisyNet |        Yes        | 0.7004 |  77.18%  | 899.34s (179.87s for 20 epochs) |

The same model as before is used, and the Data Augmentation set up as the following:

- RandomAffine(0, scale=(.2, 1.2), shear=10)
- RandomHorizontalFlip()
- RandomRotation(10)

A sample of the augmented dataset images:

<img src="plots/ex3/pytorch/dataset_augmented.png" height="180" />


#### Loss and Accuracy Plots

<img src="plots/ex3/pytorch/praisynet_loss.png" height="240" />
<img src="plots/ex3/pytorch/praisynet_accuracy.png" height="240" />

There is clearly no overfitting anymore. The validation accuracy follows the training accuracy even up to 100 epochs.

#### Confusion Matrix

<img src="plots/ex3/pytorch/praisynet_confusion_matrix.png" height="400" />

The confusion matrix is still good. So the results are not changed, the model only learns better.

#### 10 Worst Classified Images

| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                        Image                         |
| :---: | :--------: | :--------: | :-------: | :--------------------------------------------------: |
|  10   |    365     |   Plane    |   Deer    | <img src="ten_worst/ex3/pytorch/praisynet/10.png" /> |
|   9   |    6901    |   Plane    |   Bird    | <img src="ten_worst/ex3/pytorch/praisynet/9.png" />  |
|   8   |    8698    |    Deer    |   Frog    | <img src="ten_worst/ex3/pytorch/praisynet/8.png" />  |
|   7   |    1118    |   Horse    |   Deer    | <img src="ten_worst/ex3/pytorch/praisynet/7.png" />  |
|   6   |    8344    |    Deer    |   Horse   | <img src="ten_worst/ex3/pytorch/praisynet/6.png" />  |
|   5   |    9132    |    Deer    |   Bird    | <img src="ten_worst/ex3/pytorch/praisynet/5.png" />  |
|   4   |    453     |    Dog     |    Cat    | <img src="ten_worst/ex3/pytorch/praisynet/4.png" />  |
|   3   |    115     |   Horse    |    Cat    | <img src="ten_worst/ex3/pytorch/praisynet/3.png" />  |
|   2   |    6786    |    Dog     |    Cat    | <img src="ten_worst/ex3/pytorch/praisynet/2.png" />  |
|   1   |    4571    |   Horse    |   Deer    | <img src="ten_worst/ex3/pytorch/praisynet/1.png" />  |

<br /><br />

## 4. Transfer learning / Fine-tuning on CIFAR10 dataset

For this part, we are going to use ResNet50 pre-trained on ImageNet. 
We want to specify our input shape and remove the classifier to add our own so the model can classify 10 classes.

#### Model Summaries

Now that we know data augmentation helps improve results, we want to try fine-tuning with and without data augmentation.

|     ID     | Data Augmentation |  Loss  | Accuracy | Training time |
| :--------: | :---------------: | :----: | :------: | :-----------: |
|  MyResNet  |        No         | 1.6432 |  82.43%  |   1076.82s    |
| MyResNetDA |        Yes        | 0.8252 |  80.67%  |   1079.93s    |

<br />

For both MyResnet and MyResNetDA, the accuracy has increased of more than 10% compared to PraisyNet (69.80%), which is a quite good improvement (even though the training time also increased). 

<br />

#### Loss, Accuracy Plots and Confusion Matrices

|     ID     |                            Loss Plot                             |                            Accuracy Plot                             |                               Confusion Matrix                               |
| :--------: | :--------------------------------------------------------------: | :------------------------------------------------------------------: | :--------------------------------------------------------------------------: |
|  MyResNet  |  <img src="plots/ex4/pytorch/MyResNet/loss.png" height="150" />  |  <img src="plots/ex4/pytorch/MyResNet/accuracy.png" height="150" />  |  <img src="plots/ex4/pytorch/MyResNet/confusion_matrix.png" height="150" />  |
| MyResNetDA | <img src="plots/ex4/pytorch/MyResNetDA/loss.png" height="150" /> | <img src="plots/ex4/pytorch/MyResNetDA/accuracy.png" height="150" /> | <img src="plots/ex4/pytorch/MyResNetDA/confusion_matrix.png" height="150" /> |

<br />

Despite better accuracies in both models, their respective plots are questionable.
Indeed, for MyResNet, the accuracy and loss plots show overfitting around the third epoch.
Fortunately (or not), we can add data augmentation in order to solve our problem. The thing is, we now observe on the accuracy plot a train accuracy that is above the test accuracy. This is a sign of underfitting, due to data augmentation. We might be adding and changing input data so much that the model become very efficient on the “simple” data in the test dataset (because it's not transformed), but still has trouble giving a good result on a lot of different data.  

<br />

#### 10 Worst Classified Images

|       |  MyResNet  |            |           |                                                     | MyResNetDA |            |           |                                                       |
| :---: | :--------: | :--------: | :-------: | :-------------------------------------------------: | :--------: | :--------: | :-------: | :---------------------------------------------------: |
| Rank  | Image Idx. | Pred. Cat. | Act. Cat. |                        Image                        | Image Idx. | Pred. Cat. | Act. Cat. |                         Image                         |
|  10   |    8033    |    Dog     |    Cat    | <img src="ten_worst/ex4/pytorch/MyResNet/10.png" /> |    9336    |    Dog     |    Cat    | <img src="ten_worst/ex4/pytorch/MyResNetDA/10.png" /> |
|   9   |    9542    |   Truck    |    Car    | <img src="ten_worst/ex4/pytorch/MyResNet/9.png" />  |    3390    |    Dog     |    Cat    | <img src="ten_worst/ex4/pytorch/MyResNetDA/9.png" />  |
|   8   |    345     |    Cat     |    Dog    | <img src="ten_worst/ex4/pytorch/MyResNet/8.png" />  |    4248    |    Dog     |    Cat    | <img src="ten_worst/ex4/pytorch/MyResNetDA/8.png" />  |
|   7   |    1731    |    Cat     |    Dog    | <img src="ten_worst/ex4/pytorch/MyResNet/7.png" />  |    8069    |    Bird    |   Deer    | <img src="ten_worst/ex4/pytorch/MyResNetDA/7.png" />  |
|   6   |    4930    |   Truck    |    Car    | <img src="ten_worst/ex4/pytorch/MyResNet/6.png" />  |    6383    |    Dog     |    Cat    | <img src="ten_worst/ex4/pytorch/MyResNetDA/6.png" />  |
|   5   |    5598    |   Truck    |    Car    | <img src="ten_worst/ex4/pytorch/MyResNet/5.png" />  |    9406    |    Cat     |    Dog    | <img src="ten_worst/ex4/pytorch/MyResNetDA/5.png" />  |
|   4   |    8549    |   Truck    |    Car    | <img src="ten_worst/ex4/pytorch/MyResNet/4.png" />  |    5724    |    Cat     |    Dog    | <img src="ten_worst/ex4/pytorch/MyResNetDA/4.png" />  |
|   3   |    4206    |   Truck    |    Car    | <img src="ten_worst/ex4/pytorch/MyResNet/3.png" />  |    6174    |    Dog     |    Cat    | <img src="ten_worst/ex4/pytorch/MyResNetDA/3.png" />  |
|   2   |    601     |    Cat     |    Dog    | <img src="ten_worst/ex4/pytorch/MyResNet/2.png" />  |    5237    |    Bird    |   Deer    | <img src="ten_worst/ex4/pytorch/MyResNetDA/2.png" />  |
|   1   |    3067    |    Dog     |    Cat    | <img src="ten_worst/ex4/pytorch/MyResNet/1.png" />  |    6237    |    Dog     |    Cat    | <img src="ten_worst/ex4/pytorch/MyResNetDA/1.png" />  |

By analysing each ranking, we can observe that
- MyResNet tends to mistake Car as Truck (5 times), Cat as Dog (2 times) and Dog as Cat (3 times)
- MyResNetDA tends to mistake Cat as Dog (6 times) and Dog as Cat (2 times)

As mentionned earlier, since those classes are of "same type", the error is understandable.
