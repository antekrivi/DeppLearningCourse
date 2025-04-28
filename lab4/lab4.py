import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

# Postavke za TensorBoard
writer = SummaryWriter("runs/emnist_letters")

# Transformacija slika
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Učitavanje EMNIST skupa podataka, obavezno koristiti split='letters'
trainset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
testset = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

# Učitavanje podataka u batch-evima
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Dimenzije mreže
input_size = 28 * 28
hidden_size_1 = 256
hidden_size_2 = 128
output_size = 26  # EMNIST slova: 26 klasa (A-Z)

# Kreiranje modela
model = nn.Sequential(
    nn.Linear(input_size, hidden_size_1),
    nn.ReLU(),
    nn.Linear(hidden_size_1, hidden_size_2),
    nn.ReLU(),
    nn.Linear(hidden_size_2, output_size),
    nn.LogSoftmax(dim=1)
)

# Funkcija greške i optimizator
loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Treniranje modela
epochs = 10
time0 = time()

for e in range(epochs):
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()

        output = model(images)
        loss = loss_fn(output, labels - 1)  # EMNIST Letters ima labelu 1-26, ne 0-25
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == (labels - 1)).sum().item()

    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f"Epoch {e+1} - Training loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
    writer.add_scalar("Training Loss", avg_loss, e+1)
    writer.add_scalar("Training Accuracy", train_accuracy, e+1)

print("\nTraining completed in {:.2f} minutes".format((time() - time0) / 60))

# Evaluacija na test skupu
correct = 0
total = 0
running_test_loss = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.shape[0], -1)
        output = model(images)
        loss = loss_fn(output, labels - 1)
        running_test_loss += loss.item()

        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == (labels - 1)).sum().item()

test_loss = running_test_loss / len(test_loader)
test_accuracy = 100 * correct / total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
writer.add_scalar("Test Loss", test_loss)
writer.add_scalar("Test Accuracy", test_accuracy)
writer.close()
