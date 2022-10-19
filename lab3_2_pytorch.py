# Neural network with hidden layer for binary classification using Keras
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

DIGIT = 5
BATCH_SIZE = 32
EPOCHS = 40
HL_UNITS = []


class NeuralNetwork(nn.Module):
    """TODO: Modify NN (add hidden layer)."""

    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


if __name__ == "__main__":
    # Set up the device.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scores = []
    durations = []
    for hlu in HL_UNITS:
        print(f"\n###\n### HL UNITS: {hlu}\n###")

        # Load the data.
        training_data = datasets.MNIST(root="data", train=True, transform=ToTensor(), download=True)
        test_data = datasets.MNIST(root="data", train=False, transform=ToTensor(), download=True)
        train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        test_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)

        # Retrieve image size.
        image_size = 0
        for x, y in train_dataloader:
            image_size = x.shape[2] * x.shape[3]
            break
        assert image_size != 0

        # Define the model and its parameters.
        model = NeuralNetwork(image_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()

        # Train the model.
        history = []
        training_start_time = time.time()
        for e in range(EPOCHS):
            print(f"Epoch {e + 1}/{EPOCHS}")
            dataset_size = len(train_dataloader.dataset)  # type: ignore
            num_batches = len(train_dataloader)
            epoch_loss, epoch_accuracy = 0, 0
            model.train()

            epoch_start_time = time.time()
            for x, y in train_dataloader:
                # Flatten x, set up y for binary classification, and initialize the device.
                x = x.reshape(-1, image_size)
                y = torch.where(y == DIGIT, 1, 0)
                y = y.unsqueeze(-1).to(torch.float32)
                x, y = x.to(device), y.to(device)

                # Achieve the forward pass and compute the error.
                pred = model(x)
                loss = loss_fn(pred, y)

                # Achieve the backpropagation.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Apply the batch metrics to the epoch metrics.
                epoch_loss += loss.item() / num_batches
                epoch_accuracy += (torch.where(pred >= .5, 1, 0) == y).sum().float() / BATCH_SIZE / num_batches
            epoch_time = time.time() - epoch_start_time

            # Save and display epoch results.
            history += [(epoch_loss, epoch_accuracy)]
            print(f"{epoch_time:.0f}s - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f}")
        training_time = time.time() - training_start_time

        # Evaluate the model.
        dataset_size = len(test_dataloader.dataset)  # type: ignore
        num_batches = len(test_dataloader)
        model_loss, model_accuracy = 0, 0
        model.eval()
        for x, y in test_dataloader:
            # Flatten x, set up y for binary classification, and initialize the device.
            x = x.reshape(-1, image_size)
            y = torch.where(y == DIGIT, 1, 0)
            y = y.unsqueeze(-1).to(torch.float32)
            x, y = x.to(device), y.to(device)

            # Predict the class.
            pred = model(x)
            loss = loss_fn(pred, y)

            # Apply the batch metrics to the model metrics.
            model_loss += loss.item() / num_batches
            model_accuracy += (torch.where(pred >= .5, 1, 0) == y).sum().float() / BATCH_SIZE / num_batches
        print(f"loss: {model_loss:.4f} - accuracy: {model_accuracy:.4f}")
        scores += [(model_loss, model_accuracy)]
        durations += [training_time]

        # Display the summary.
        print(f"SUMMARY FOR HL UNITS {hlu}:\n    - Training Time: {training_time:.0f}s\n    - Loss: {model_loss:.2f}\n    - Accuracy: {model_accuracy:.2f}")

    # Plot the loss history.
    plt.clf()
    plt.xscale("log")
    plt.plot(HL_UNITS, np.array(scores)[:, 0])
    plt.xlabel("HL Units")
    plt.ylabel("Loss")
    plt.title("Loss over HL Units")
    plt.savefig("plots/ex1/lab3_2_pytorch_bs_cmp_loss.png")

    # Plot the accuracy history.
    plt.clf()
    plt.xscale("log")
    plt.plot(HL_UNITS, np.array(scores)[:, 1])
    plt.xlabel("HL Units")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over HL Units")
    plt.savefig("plots/ex1/lab3_2_pytorch_bs_cmp_accuracy.png")

    # Plot the duration history.
    plt.clf()
    plt.xscale("log")
    plt.plot(HL_UNITS, durations)
    plt.xlabel("HL Units")
    plt.ylabel("Duration")
    plt.title("Duration over HL Units")
    plt.savefig("plots/ex1/lab3_2_pytorch_bs_cmp_duration.png")
