import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = 10
CHANNELS = 1

EPOCHS = 20
BATCH_SIZE = 32

# Convolution Layers: Array of tuples (in_channels, out_channels, kernel_size, stride, padding).
CONVS = [(CHANNELS, 32, 3, 1, 'valid')]

# Max Pooling Layers: Array of tuple (kernel_size, stride, padding).
MAXPOOLS = []

# Linear Layers: Array of tuples (in_features, out_features, activation_function).
LINEARS = [(21632, CLASSES, 'softmax')]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Define the convolution layers.
        self.convs = nn.ParameterList()
        for in_channels, out_channels, kernel_size, stride, padding in CONVS:
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

        # Define the maximum pooling layers.
        self.maxpools = nn.ParameterList()
        for kernel_size, stride, padding in MAXPOOLS:
            self.maxpools.append(nn.MaxPool2d(kernel_size, stride, padding))

        # Define the flatten layer.
        self.flatten = nn.Flatten()

        # Define the linear layers.
        self.linears = nn.ParameterList()
        for in_features, out_features, act_fn in LINEARS:
            self.linears.append(nn.Linear(in_features, out_features))

        # Define the activation functions.
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        # Apply the convolution layers.
        for conv in self.convs:
            x = conv(x)

        # Apply the maximum pooling layers.
        for maxpool in self.maxpools:
            x = maxpool(x)

        # Apply the flatten layer.
        x = self.flatten(x)

        # Apply the linear layers.
        i = 0
        for linear in self.linears:
            if LINEARS[i][2] == 'sigmoid':
                act_fn = self.sigmoid
            if LINEARS[i][2] == 'relu':
                act_fn = self.relu
            if LINEARS[i][2] == 'tanh':
                act_fn = self.tanh
            if LINEARS[i][2] == 'softmax':
                act_fn = self.softmax
            x = act_fn(linear(x))
            i += 1

        return x


if __name__ == '__main__':
    # Set up device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the data.
    training_data = datasets.MNIST(root='data', train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]), download=True)
    test_data = datasets.MNIST(root='data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]), download=True)

    # Retrieve the data loaders.
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)

    # Define the model.
    model = CNN().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Train the model.
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    dataset_size = len(train_dataloader.dataset)  # type: ignore
    num_batches = len(train_dataloader)
    training_start_time = time.time()
    model.train()
    for e in range(EPOCHS):
        print(f'Epoch {e + 1}/{EPOCHS}')
        epoch_start_time = time.time()
        epoch_loss, epoch_accuracy = 0, 0
        epoch_val_loss, epoch_val_accuracy = 0, 0

        # actual training
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / num_batches
            epoch_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item() / BATCH_SIZE / num_batches

        # validation
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            epoch_val_loss += loss.item() / num_batches
            epoch_val_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item() / BATCH_SIZE / num_batches

        epoch_time = time.time() - epoch_start_time
        history['loss'] += [epoch_loss]
        history['accuracy'] += [epoch_accuracy]
        history['val_loss'] += [epoch_val_loss]
        history['val_accuracy'] += [epoch_val_accuracy]
        print(f'{epoch_time:.2f}s - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f} - val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_accuracy:.4f}')
    training_time = time.time() - training_start_time

    # Evaluate the model.
    dataset_size = len(test_dataloader.dataset)  # type: ignore
    num_batches = len(test_dataloader)
    evaluating_start_time = time.time()
    model.eval()
    model_loss, model_accuracy = 0, 0
    y_pred, y_true = [], []
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        model_loss += loss.item() / num_batches
        model_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item() / BATCH_SIZE / num_batches
        pred = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
        y_pred.extend(pred)
        y = y.data.cpu().numpy()
        y_true.extend(y)
    evaluating_time = time.time() - evaluating_start_time
    print(f'{evaluating_time:.2f}s - loss: {model_loss:.4f} - accuracy: {model_accuracy:.4f}')

    # Display the summary.
    print(f"SUMMARY:\n    - Loss: {model_loss:.4f}\n    - Accuracy: {model_accuracy:.4f}\n    - Training Time: {training_time:.2f}s")

    # Plot the loss.
    plt.plot(history['loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epoch')
    plt.savefig('plots/ex1/pytorch/model_1_loss.png')
    plt.clf()

    # Plot the accuracy.
    plt.plot(history['accuracy'], label='Training')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epoch')
    plt.savefig('plots/ex1/pytorch/model_1_accuracy.png')
    plt.clf()

    # Plot the confusion matrix.
    cm = confusion_matrix(y_true, y_pred)
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    df_cm = pd.DataFrame(cm/np.sum(cm) * 10, index=[i for i in labels], columns=[i for i in labels])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig("plots/ex1/pytorch/model_1_confusion_matrix.png")
    plt.clf()
