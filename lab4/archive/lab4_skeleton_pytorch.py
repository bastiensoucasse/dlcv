import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# Get cpu or gpu device for training.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

training_data = datasets.MNIST(root="./data", download=True,
                                      train=True, 
                                      transform=transforms.ToTensor())

test_data = datasets.MNIST(root="./data", download=True,
                                  train=False,
                                  transform=transforms.ToTensor())
# data is in [0; 1] (thanks to ToTensor()), but there is no "standardisation"


batch_size = 32
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(training_data, batch_size=batch_size)
