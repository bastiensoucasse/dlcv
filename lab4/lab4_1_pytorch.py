import time

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MODEL = 'model4'

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NUM_CLASSES = len(CLASSES)
NUM_CHANNELS = 1

BATCH_SIZE = 32
NUM_EPOCHS = 20

PLOT = True


class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.conv = nn.Conv2d(NUM_CHANNELS, 32, 3, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(21632, NUM_CLASSES)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        # x = self.softmax(x)
        return x


class model2(nn.Module):
    def __init__(self):
        super(model2, self).__init__()
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 64, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 32, 3, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 16, 3, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1600, NUM_CLASSES)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear(x)
        # x = self.softmax(x)
        return x


class model3(nn.Module):
    def __init__(self):
        super(model3, self).__init__()
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 64, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 32, 3, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 16, 3, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1600, NUM_CLASSES)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear(x)
        # x = self.softmax(x)
        return x


class model4(nn.Module):
    def __init__(self):
        super(model4, self).__init__()
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 64, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 32, 3, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 16, 3, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1600, 128)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(128, NUM_CLASSES)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        # x = self.softmax(x)
        return x


if __name__ == '__main__':
    # Set up device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}.")

    # Load the data.
    train_dataset = datasets.MNIST('data', train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]), download=True)
    test_dataset = datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]), download=True)
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Define the model.
    model = eval(MODEL + '()').to(device)
    print(f"Model: {MODEL}.")
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train the model.
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    training_start_time = time.time()
    for e in range(NUM_EPOCHS):
        print(f'Epoch {e + 1}/{NUM_EPOCHS}')
        epoch_start_time = time.time()

        # training
        model.train()
        running_loss, running_accuracy = 0, 0
        for x, y in train_data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / len(train_data_loader)
            running_accuracy += (pred.argmax(1) == y).sum().item() / BATCH_SIZE / len(train_data_loader)

        # validation
        with torch.no_grad():
            model.eval()
            running_val_loss, running_val_accuracy = 0, 0
            for x, y in test_data_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                running_val_loss += loss.item() / len(test_data_loader)
                running_val_accuracy += (pred.argmax(1) == y).sum().item() / BATCH_SIZE / len(test_data_loader)

        epoch_time = time.time() - epoch_start_time
        history['loss'] += [running_loss]
        history['accuracy'] += [running_accuracy]
        history['val_loss'] += [running_val_loss]
        history['val_accuracy'] += [running_val_accuracy]
        print(f'{epoch_time:.2f}s - loss: {running_loss:.4f} - accuracy: {running_accuracy:.4f} - val_loss: {running_val_loss:.4f} - val_accuracy: {running_val_accuracy:.4f}')
    training_time = time.time() - training_start_time

    # Evaluate the model.
    with torch.no_grad():
        y_pred, y_true = [], []
        evaluating_start_time = time.time()
        model.eval()
        running_loss, running_accuracy = 0, 0
        for x, y in test_data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            running_loss += loss.item() / len(test_data_loader)
            running_accuracy += (pred.argmax(1) == y).sum().item() / BATCH_SIZE / len(test_data_loader)
            y_pred.extend(pred.argmax(1).cpu())
            y_true.extend(y.cpu())
        evaluating_time = time.time() - evaluating_start_time
        print(f'{evaluating_time:.2f}s - loss: {running_loss:.4f} - accuracy: {running_accuracy:.4f}')

    # Display the summary.
    print(f'SUMMARY:\n    - Loss: {running_loss:.4f}\n    - Accuracy: {running_accuracy:.4f}\n    - Training Time: {training_time:.2f}s')

    if not PLOT:
        exit()

    # Plot the loss.
    plt.plot(history['loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epoch')
    plt.savefig('plots/ex1/pytorch/%s_loss.png' % MODEL)
    plt.clf()

    # Plot the accuracy.
    plt.plot(history['accuracy'], label='Training')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epoch')
    plt.savefig('plots/ex1/pytorch/%s_accuracy.png' % MODEL)
    plt.clf()

    # Plot the confusion matrix.
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap=plt.cm.magma)
    plt.savefig('plots/ex1/pytorch/%s_confusion_matrix.png' % MODEL)
    plt.clf()
