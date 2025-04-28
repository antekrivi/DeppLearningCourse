import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Postavke za uređaj
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')
print(f"Using device: {device}")

# Inicijalizacija writera za TensorBoard
writer = SummaryWriter(f'runs/emnist_letters/{datetime.now().strftime("%Y%m%d-%H%M%S")}')

# Transformacija slika
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Učitavanje EMNIST skupa podataka
trainset = datasets.EMNIST(root='./emnist_data', split='letters', train=True, download=True, transform=transform)
testset = datasets.EMNIST(root='./emnist_data', split='letters', train=False, download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=True)

# Dimenzije mreže
input_size = 28 * 28
hidden_size_1 = 256
hidden_size_2 = 128
output_size = 26  # EMNIST slova: 26 klasa (A-Z)

# Kreiranje modela
model = nn.Sequential(
    nn.Linear(input_size, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, output_size),
    nn.LogSoftmax(dim=1)
).to(device)

# Funkcija greške i optimizator
loss_fn = nn.NLLLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epochs = 10
train_per_epoch = int(len(trainset) / 64)  # broj batch-eva po epohi

for e in range(epochs):
    running_loss = 0
    correct = 0
    total = 0

    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, ncols=100)
    for idx, (images, labels) in loop:
        images, labels = images.to(device), labels.to(device)

        # Preoblikovanje slika
        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()

        # Model
        output = model(images)
        loss = loss_fn(output, labels - 1)  # EMNIST Letters ima labelu 1-26, ne 0-25
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Izračunavanje tačnosti
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == (labels - 1)).sum().item()

        # Prikaz napretka i gubitka
        accuracy = 100 * correct / total
        loop.set_description(f"Epoch [{e+1}/{epochs}]")
        loop.set_postfix(loss=running_loss / (idx + 1), acc=accuracy)
        
        # Dodavanje podataka u TensorBoard
        writer.add_scalar('Loss/train', running_loss / (idx + 1), (e * train_per_epoch) + idx)
        writer.add_scalar('Accuracy/train', accuracy, (e * train_per_epoch) + idx)

    # Kraj epohe
    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f"Epoch {e+1} - Training loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
    writer.add_scalar("Loss/train_epoch", avg_loss, e + 1)
    writer.add_scalar("Accuracy/train_epoch", train_accuracy, e + 1)

# Evaluacija na test skupu
correct = 0
total = 0
running_test_loss = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
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
writer.add_scalar("Loss/test", test_loss)
writer.add_scalar("Accuracy/test", test_accuracy)

# Zatvaranje TensorBoard writer-a
writer.close()
