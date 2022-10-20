# Single neuron neural network for binary classification using PyTorch

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

DIGIT = 5
EPOCHS = 40
BATCH_SIZES = [60000, 2048, 1024, 512, 256, 128, 64, 32, 16]
NUM_BS = len(BATCH_SIZES)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.output = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.output(x))


if __name__ == "__main__":
    # Set up the device.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Array of history over model version (one history being an array of (training_loss, training_accuracy) over epoch).
    hists = []

    # Array of evaluation score over model version (one evaluation score being (evaluation_loss, evaluation_score)).
    scores = []

    # Array of training time over model version.
    durations = []

    for bs in BATCH_SIZES:
        print(f"\n###\n### BATCH SIZE: {bs}\n###")

        # Load the data.
        training_data = datasets.MNIST(root="data", train=True, transform=ToTensor(), download=True)
        test_data = datasets.MNIST(root="data", train=False, transform=ToTensor(), download=True)
        train_dataloader = DataLoader(training_data, batch_size=bs, shuffle=True, num_workers=2)
        test_dataloader = DataLoader(training_data, batch_size=bs)

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
                epoch_accuracy += (torch.where(pred >= .5, 1, 0) == y).sum().float() / bs / num_batches
            epoch_time = time.time() - epoch_start_time

            # Save and display epoch results.
            history += [(epoch_loss, epoch_accuracy)]
            print(f"{epoch_time:.2f}s - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f}")
        training_time = time.time() - training_start_time

        # At the end of the model training, save its history (loss & accuracy over epoch) into the array of all history (one for each model version).
        hists += [history]

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
            model_accuracy += (torch.where(pred >= .5, 1, 0) == y).sum().float() / bs / num_batches
        print(f"loss: {model_loss:.4f} - accuracy: {model_accuracy:.4f}")
        scores += [(model_loss, model_accuracy)]
        durations += [training_time]

        # Display the summary.
        print(f"SUMMARY FOR BATCH SIZE {bs}:\n    - Loss: {model_loss:.4f}\n    - Accuracy: {model_accuracy:.4f}\n    - Training Time: {training_time:.2f}s")

    # Plot Training Loss Over Epoch
    plt.clf()
    for i in range(NUM_BS):
        plt.plot(np.array(hists)[i, :, 0], label=f"Batch Size: {BATCH_SIZES[i]}")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epoch")
    plt.savefig("plots/ex1/pytorch/loss_over_epoch.png")

    # Plot Training Accuracy Over Epoch
    plt.clf()
    for i in range(NUM_BS):
        plt.plot(np.array(hists)[i, :, 1], label=f"Batch Size: {BATCH_SIZES[i]}")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epoch")
    plt.savefig("plots/ex1/pytorch/accuracy_over_epoch.png")

    # Plot Evaluation Loss Over BS
    plt.clf()
    plt.plot(BATCH_SIZES, np.array(scores)[:, 0])
    plt.xlabel("Batch Size")
    plt.ylabel("Loss")
    plt.title("Loss over Batch Size")
    plt.savefig("plots/ex1/pytorch/loss_over_bs.png")

    # Plot Evaluation Accuracy Over BS
    plt.clf()
    plt.plot(BATCH_SIZES, np.array(scores)[:, 1])
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Batch Size")
    plt.savefig("plots/ex1/pytorch/accuracy_over_bs.png")

    # Plot Training Time Over BS
    plt.clf()
    plt.plot(BATCH_SIZES, durations)
    plt.xlabel("Batch Size")
    plt.ylabel("Training Time")
    plt.title("Training Time over Batch Size")
    plt.savefig("plots/ex1/pytorch/training_time_over_bs.png")
