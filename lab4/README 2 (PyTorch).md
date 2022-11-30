## 3. Data Augmentation

### Model Without Data Augmentation

We are now going to use a stable model based on `model5` that we previously saw, and matching the one we used on the Keras version.

#### Model Summary

|    ID     | Data Augmentation |  Loss  | Accuracy | Training Time |
| :-------: | :---------------: | :----: | :------: | :-----------: |
| PraisyNet |        No         | 1.1851 |  69.80%  |    160.63s    |

[architecture]

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

[architecture]

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
Fortunately (or not), we can add data augmentation in order to solve our problem. The thing is, we now observe on the accuracy plot a train accuracy that is above the test accuracy. This is a sign of underfitting, due to data augmentation [TODO: add some details].

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
- MyResNet gives the worst results when mistaking Car as Truck (5 times), Cat as Dog (2 times) and Dog as Cat (3 times)
- MyResNetDA gives the worst results when mistaking Cat as Dog (6 times) and Dog as Cat (2 times)

As mentionned earlier, since those classes are of "same type", the error is understandable.