# Single neuron neural network for binary classification using Pytorch

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 32
EPOCHS = 40


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.linear(x))


if __name__ == "__main__":
    # Set Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load Data
    training_data = datasets.MNIST(root="./data", download=True, train=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root="./data", download=True, train=False, transform=transforms.ToTensor())
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    for x, y in test_dataloader:
        x_shape = x.shape
        y_shape = y.shape
        print(f"Shape of X [N, C, H, W]: {x.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Define Model
    input_size = 0

    model = NeuralNetwork(input_size).to(device)
    print(model)

    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    loss_function = nn.CrossEntropyLoss

    searched_digit = 5

    # Train Model
    for e in range(EPOCHS):
        size = len(train_dataloader.dataset)
        num_batches = len(train_dataloader)

        running_acc = 0
        running_loss = 0
        for batch, (x, y) in enumerate(train_dataloader):
            x = x.reshape(-1, input_size)
            y = torch.where(y == searched_digit, 1, 0)
            y = y.unsqueeze(-1).to(torch.float32)
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_function(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_acc = 0 # TODO
            running_loss = 0 # TODO

    loss = 0 # TODO
    accuracy = 0 # TODO
    print(f"Epoch {e}/{EPOCHS}: loss={loss:.4f} acc={accuracy:.4f}")
