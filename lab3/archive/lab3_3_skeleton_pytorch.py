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

for X, y in test_dataloader:
    X_shape = X.shape
    y_shape = y.shape
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# TODO: Complete this code

# input_size = 
# num_classes = 
# hidden_size = 

# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#       #TODO        
        
#     def forward(self, x):
#         #TODO


# model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)
# print(model)

# lr = 0.001
# loss_fn = 
# optimizer = 

# # Train
# epochs = 100
# for e in range(epochs):
#     size = len(train_dataloader.dataset)
#     num_batches = len(train_dataloader)
    
#     running_acc = 0
#     running_loss = 0
#     for batch, (X, y) in enumerate(train_dataloader):

#         X = # flatten
        
#         X, y = X.to(device), y.to(device)

#         #forward pass & loss
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         #backward pass / backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_acc =  #TODO
        
#         running_loss = #TODO
#     loss = #TODO
#     accuracy = #TODO
#     print(f'{e}/{epochs}: loss={loss:.4f} acc={accuracy:.4f}')

    
# # Evaluate
# TODO
