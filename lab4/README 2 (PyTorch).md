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
