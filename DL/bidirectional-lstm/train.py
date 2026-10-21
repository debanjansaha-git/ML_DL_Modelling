"""
@Author: Debanjan Saha
@Date: 22 Nov, 2021
@Description: Bidirectional LSTM on MNIST dataset using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from model import BLSTM
from utils import check_accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Hyper-parameters
INPUT_SIZE = 28
SEQUENCE_LEN = 28
NUM_LAYERS = 2
HIDDEN_SIZE = 256
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 10
MODEL_PATH = "models/"

### Load Data
train_dataset = datasets.MNIST(
    root='./dataset/', train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root='./dataset/', train=False, transform=transforms.ToTensor(), download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)


### Initialize Model
model = BLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)

### Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train():
    ### Train Model
    for epoch in tqdm(range(NUM_EPOCHS)):
        print() # blank line
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device).squeeze(1)
            targets = targets.to(device)

            #### Forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            #### Backward pass
            optimizer.zero_grad()
            loss.backward()

            #### Gradient Descent
            optimizer.step()

            #### Print
            if batch_idx % 100 == 0:
                print(f"Epoch: [{epoch+1}/{NUM_EPOCHS}], Batch: [{batch_idx}/{len(train_loader)}], Loss: {loss:.4f}")


if __name__ == "__main__":
    train()
    check_accuracy(train_loader, model) # Correct Results: 59720 / 60000,    Accuracy: 99.53%
    check_accuracy(test_loader, model) # Correct Results: 9894 / 10000,    Accuracy: 98.94%
    # torch.save(model, MODEL_PATH)