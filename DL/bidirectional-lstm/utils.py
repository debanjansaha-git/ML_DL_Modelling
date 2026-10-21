"""
@Author: Debanjan Saha
@Date: 22 Nov, 2021
@Description: Function to check accuracy
"""
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Correct Results: {num_correct} / {num_samples},    Accuracy: {float(num_correct)/float(num_samples)*100:.2f}%")