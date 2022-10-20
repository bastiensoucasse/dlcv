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
HL_UNITS = [8, 16, 32, 64, 128]
NB_HLU = len(HL_UNITS)
ACTIVATION_FUN = ["sigmoid", "relu", "tanh"]
NB_AF = len(ACTIVATION_FUN)

class NeuralNetwork(nn.Module):
    def getAF(af):
        if af == "sigmoid":
            return nn.Sigmoid()
        elif af == "relu":
            return nn.ReLU()
        elif af == "tanh":
            return nn.Tanh()
        else:
            print("Invalid activation function!\n")
            exit()

    def __init__(self, input_size, hlu, af1, af2):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hlu)
        self.output = nn.Linear(hlu, 1)
        self.af1 = self.getAF(af1)
        self.af2 = self.getAF(af2)

    def forward(self, x):
        return self.af2(self.output(self.af1(self.hidden(x))))


if __name__ == "__main__":
    # Set up the device.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Array of hists (...)
    HIST = []

    # Array of history over model version (one history being an array of (training_loss, training_accuracy) over epoch).
    # hists = []

    # Array of evaluation score over model version (one evaluation score being (evaluation_loss, evaluation_score)).
    scores = []

    # Array of training time over model version.
    durations = []

    # for hlu in HL_UNITS:
    #     print(f"\n###\n### HL UNITS: {hlu}\n###")
    for af1 in ACTIVATION_FUN:
        hists = []
        for af2 in ACTIVATION_FUN:
            print(f"\n###\n### ACTIVATION FUNCTIONS: {af1, af2}\n###")

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
            # model = NeuralNetwork(image_size, hlu, "sigmoid", "sigmoid").to(device)
            model = NeuralNetwork(image_size, 64, af1, af2).to(device)
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
                model_accuracy += (torch.where(pred >= .5, 1, 0)== y).sum().float() / BATCH_SIZE / num_batches
            print(f"loss: {model_loss:.4f} - accuracy: {model_accuracy:.4f}")
            scores += [(model_loss, model_accuracy)]
            durations += [training_time]

            # Display the summary.
            # print(f"SUMMARY FOR HL UNITS {hlu}:\n    - Training Time: {training_time:.2f}s\n    - Loss: {model_loss:.4f}\n    - Accuracy: {model_accuracy:.4f}")
            print(f"SUMMARY FOR ACTIVATION FUNCTIONS {af1, af2}:\n    - Training Time: {training_time:.2f}s\n    - Loss: {model_loss:.4f}\n    - Accuracy: {model_accuracy:.4f}")
        HIST += [hists]

    # Plot Training Loss Over Epoch
    # plt.clf()
    # for i in range(NB_HLU):
    #     plt.plot(np.array(hists)[i, :, 0], label=f"{HL_UNITS[i]} units")
    # ...
    # plt.savefig("plots/ex2/pytorch/hlu/loss_over_epoch.png")
    for i in range(NB_AF):
        plt.clf()
        for j in range(NB_AF):
            # TODO: plt.plot(np.array(hists)[i, :, 0], label=f"{HL_UNITS[i]} units") 
            ntm = i
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training loss over Epoch")
        plt.savefig("plots/ex2/pytorch/af/loss_over_epoch.png")
            

    # Plot Training Accuracy Over Epoch
    # plt.clf()
    # for i in range(NB_HLU):
    #     plt.plot(np.array(hists)[i, :, 1], label=f"{HL_UNITS[i]} units")
    # ...
    # plt.savefig("plots/ex2/pytorch/hlu/accuracy_over_epoch.png")
    for i in range(NB_AF):
        plt.clf()
        for j in range(NB_AF):
            # TODO: plt.plot(np.array(hists)[i, :, 1], label=f"{HL_UNITS[i]} units")
            ntm = i
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy over Epoch")
    plt.savefig("plots/ex2/pytorch/af/accuracy_over_epoch.png")
    
    # Plot Evaluation Loss Over HLU
    # plt.clf()
    # plt.plot(HL_UNITS, np.array(scores)[:, 0])
    # plt.xlabel("HL Units")
    # plt.ylabel("Loss")
    # plt.title("Loss over HL Units")
    # plt.savefig("plots/ex2/pytorch/hlu/loss_over_hlu.png")

    # Plot Evaluation Accuracy...
    # ... over HLU
    # plt.clf()
    # plt.plot(HL_UNITS, np.array(scores)[:, 1])
    # plt.xlabel("HL Units")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy over HL Units")
    # plt.savefig("plots/ex2/pytorch/hlu/accuracy_over_hlu.png")

    # Plot Training Time 
    # ... over HLU
    # plt.clf()
    # plt.plot(HL_UNITS, durations)
    # plt.xlabel("HL Units")
    # plt.ylabel("Training Time")
    # plt.title("Training Time over HL Units")
    # plt.savefig("plots/ex2/pytorch/hlu/training_time_over_hlu.png")

    # ... over AF
    plt.clf()
    NP_ACTIVATION = np.array(ACTIVATION_FUN)
    models = np.transpose([np.repeat(NP_ACTIVATION, len(NP_ACTIVATION)), np.tile(NP_ACTIVATION, len(NP_ACTIVATION))])
    models = [model[0] + " & " + model[1] for model in models] 
    plt.figure(figsize=(15, 6))
    plt.plot(models, durations, 'o')
    plt.xlabel("Activation function")
    plt.ylabel("Duration")
    plt.legend()
    plt.title("Duration over Activation functions")
    plt.savefig("plots/ex2/pytorch/af/training_time_over_af.png")
