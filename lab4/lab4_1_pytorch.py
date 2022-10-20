import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 32
KERNEL_SIZE = 3

if __name__ == '__main__':
    # Set up device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the data.
    training_data = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
    test_data = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=True)

    # TODO: Standardize the data?

    # Retrieve the data loaders.
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
