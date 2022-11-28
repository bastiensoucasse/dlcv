# PROVOST Iantsa & SOUCASSE Bastien — DLCV Lab 4
This a temporary file.

# Keras

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

<br /><br />

### 3.2. Comparison

To conclude this part, let's compare our new best model (model10) with the ones of our original model (model7) that was the base.

|   ID    |                          Accuracy Plot                          | Accuracy |          Training time           |
| :-----: | :-------------------------------------------------------------: | :------: | :------------------------------: |
| model7  | <img src="plots/ex2/keras/model7_accuracy.png" height="150" />  |  68.57%  |             158.78s              |
| model10 | <img src="plots/ex3/keras/model10_accuracy.png" height="150" /> |  76.33%  | 1557.88s (311.58s for 20 epochs) |

In spite of an almost twice longer training time, model10 overperforms model7. 
Indeed, mode10 provides an accuracy that is almost 8% higher and the problem of overfitting seems solved
(even if the plots are not on the same number of epochs, we do not need more epochs to see that the test accuracy will not get higher in model7).


<br /><br />

## 4. Transfer learning / Fine-tuning on CIFAR10 dataset
<!-- 
N.B.: We used this tutorial
https://towardsdatascience.com/deep-learning-using-transfer-learning-python-code-for-resnet50-8acdfb3a2d38 -->


<!-- TODO -->


