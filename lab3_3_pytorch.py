# Neural network with hidden layer for mutliclass classification using PyTorch

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

DIGITS = 10
EPOCHS = 40
BATCH_SIZE = 32
HL_UNITS = 64
OPTIMIZERS = ["Adam", "RMSprop", "SGD"]
NB_OPT = len(OPTIMIZERS)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, opt, ouput_size):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, opt)
        self.output = nn.Linear(opt, ouput_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.softmax(self.output(self.sigmoid(self.hidden(x))))


if __name__ == "__main__":
    # Set up the device.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Array of history over model version (one history being an array of (training_loss, training_accuracy) over epoch).
    hists = []

    # Array of evaluation score over model version (one evaluation score being (evaluation_loss, evaluation_score)).
    scores = []

    # Array of training time over model version.
    durations = []

    for opt in OPTIMIZERS:
        print(f"\n###\n### OPTIMIZER: {opt}\n###")

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
        model = NeuralNetwork(image_size, HL_UNITS, DIGITS).to(device)
        if opt == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if opt == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
        if opt == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

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
                # Flatten x and initialize the device.
                x = x.reshape(-1, image_size)
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
                epoch_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item() / BATCH_SIZE / num_batches
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
            # Flatten x and initialize the device.
            x = x.reshape(-1, image_size)
            x, y = x.to(device), y.to(device)

            # Predict the class.
            pred = model(x)
            loss = loss_fn(pred, y)

            # Apply the batch metrics to the model metrics.
            model_loss += loss.item() / num_batches
            model_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item() / BATCH_SIZE / num_batches
        print(f"loss: {model_loss:.4f} - accuracy: {model_accuracy:.4f}")
        scores += [(model_loss, model_accuracy)]
        durations += [training_time]

        # Display the summary.
        print(f"SUMMARY FOR OPTIMIZER \"{opt}\":\n    - Training Time: {training_time:.2f}s\n    - Loss: {model_loss:.4f}\n    - Accuracy: {model_accuracy:.4f}")

    # Plot Training Loss Over Epoch
    plt.clf()
    for i in range(NB_OPT):
        plt.plot(np.array(hists)[i, :, 0], label=f"{OPTIMIZERS[i]} Optimizer")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epoch")
    plt.savefig("plots/ex3/pytorch/loss_over_epoch.png")

    # Plot Training Accuracy Over Epoch
    plt.clf()
    for i in range(NB_OPT):
        plt.plot(np.array(hists)[i, :, 1], label=f"{OPTIMIZERS[i]} Optimizer")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epoch")
    plt.savefig("plots/ex3/pytorch/accuracy_over_epoch.png")

    # Plot Evaluation Loss Over OPT
    plt.clf()
    plt.plot(OPTIMIZERS, np.array(scores)[:, 0], 'o')
    plt.xlabel("Optimizer")
    plt.ylabel("Loss")
    plt.title("Loss over Optimizer")
    plt.savefig("plots/ex3/pytorch/loss_over_opt.png")

    # Plot Evaluation Accuracy Over OPT
    plt.clf()
    plt.plot(OPTIMIZERS, np.array(scores)[:, 1], 'o')
    plt.xlabel("Optimizer")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Optimizer")
    plt.savefig("plots/ex3/pytorch/accuracy_over_opt.png")

    # Plot Training Time Over OPT
    plt.clf()
    plt.plot(OPTIMIZERS, durations, 'o')
    plt.xlabel("Optimizer")
    plt.ylabel("Training Time")
    plt.title("Training Time over Optimizer")
    plt.savefig("plots/ex3/pytorch/training_time_over_opt.png")
