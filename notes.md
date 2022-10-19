# DLCV - Lab3

sigmoid OK for binary classifier
for multiclass classifier, sigmoid leads to sum of probs != 1 
    -> need to use softmax on last layer
    -> softmax_j = e^(zj) / sum k in [1, K] e^(zj)

https://www.tensorflow.org/overview

0) Data loader
1) Model defining
2) Model "compiling" : loss + optimizer (way of implementing gradient descent: SGD, Adam, RMSProp...)
3) Training
4) Evaluation/Prediction

1) Defining
model = Sequential()
model.add(<layer1>, ...)
model.add(<layer2>, ...)
or
model = Sequential([<layer1>, <layer2>, ...])

Types of layer:
Dense (keras) / Linear (pytorch): Full connection (most used for us)
Dense(<nbOfNeurons>, activation='<activationFunction>' [sigmoid, relu, tanh])

Input shapes/dims applied to 1st layer (input)

2) Compiling
model.compile(loss=…, optimizer=… [, metrics=['accuracy']])
    - loss: ['categorical_crossentropy', 'binary_crossentropy']
    - optimizer: ['adam', 'sgd']
    - optional metrics for accuracy computation

3) Training
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32)
    - validation_data: Data from train set, used during the training phase to check what happens when changing hyperparameters.
    - batch_size: Use a smaller amount of data each pass (but still use all data) according to GPU

hist.history['acc']: accuracy on training data
hist.history['val_acc']: accuracy on validation data
hist.history['loss']: loss on training data
hist.history['val_loss']: loss on validation data

OVERFITTING: NN learns its data “by heart” but can't generalize to any data.
    - How to see it? Train loss curve keeps descending but not test loss curve which might ascend.
    - How to solve it? Cut training before the problem, or use validation data set.

N.B.:
    - Training Data Set: To train the model (training phase).
    - Validation Data Set: To check hyperparameters influence without breaking the training (training phase).
        Don't use x_test/y_test but do it anyway. 🙃
    - Test Data Set: To test the final model (evaluation phase).

4)
    score      = model.evaluate(x_test, y_test)
loss, accuracy = 


For pytorch:
0) Data loading
1) Model definition
2) Loss & optimizer definition
3) Training:
    for over epochs and batches
        forward pass + loss
        backpropagation

4) Model evaluation on test set



import torch
       torchvision

device = torch.device('cuda' if torch_cuda_is_available() else 'cpu')

0) Data loading
training_data = datasets.MNIST(root="./data", download=True, train=True, transform=transforms.ToTensor())
    might not work because website broken -> dept-info.labri.fr/~mansenca/DLCV2022/MNIST.tar.gz
test_data = same thing but train=False

train_loader = DataLoader(training_data, batch_size=... [, shuffle= True, num_workers=2])
test_loader = DataLoader(test_data, ...)

1) Model definition
class NeuralNet(nn.Module):
    # architecture
    def __init__(self, input_size, ...):
        super(NeuralNet, self).__init__()
        super.linear = nn.Linear(inputsize, outputsize)
        ...
        self.sig = nn.Sigmoid()

    # forward pass
    def forward(self, x):
        return self.sig(self.linear(x))
        # or this if self.sig = nn.Sigmoid() not done
        return torch.sigmoid(self.linear(x))

2) Loss & optimizer definition
loss_fn = nn.CrossEntropyLoss()
        = nn.BinaryCrossEntropyLoss() #Warning! does softmax and consider that y is not one hot encoded
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr)

3) Training
for e in range(epochs):
    for X, Y in traindataloader:
        X = X.reshape(-1, input_size) # data flattening
        X, Y = X.to(device), y.to(device)

        # forward + loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# don't know where the fuck it's supposed to be
model = NeuralNet(inputsize=784).to(device)

4) Evaluation
for X, Y in testdataloader:
    about the same shit
    acc = ...
