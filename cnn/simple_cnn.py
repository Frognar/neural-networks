import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
matplotlib.use('Qt5Agg')


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FMNISTDataset(Dataset):
    def __init__(self, inputs, targets):
        inputs = inputs.float() / 255
        inputs = inputs.view(-1, 1, 28, 28)
        self.x, self.y = inputs, targets

    def __getitem__(self, index):
        return self.x[index].to(device), self.y[index].to(device)

    def __len__(self):
        return len(self.x)


def get_model():
    model = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3200, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)
    loss_func = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=1e-3)
    return model, loss_func, opt


def train_batch(inputs, targets, model, opt, loss_func):
    model.train()
    prediction = model(inputs)
    loss = loss_func(prediction.squeeze(0), targets)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()


@torch.no_grad()
def accuracy(inputs, target, model):
    model.eval()
    prediction = model(inputs)
    max_values, argmaxes = prediction.max(-1)
    correct = argmaxes == target
    return correct.cpu().numpy().tolist()


@torch.no_grad()
def val_loss(inputs, targets, model, loss_func):
    model.eval()
    prediction = model(inputs)
    loss = loss_func(prediction, targets)
    return loss.item()


def get_data():
    train = FMNISTDataset(tr_images, tr_targets)
    trn_dataloader = DataLoader(train, batch_size=32, shuffle=True)
    val = FMNISTDataset(val_images, val_targets)
    val_dataloader = DataLoader(val, batch_size=len(val_images), shuffle=True)
    return trn_dataloader, val_dataloader


data_folder = '~/data/FMNIST'

fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
tr_images = fmnist.data
tr_targets = fmnist.targets

val_fmnist = datasets.FashionMNIST(data_folder, download=True, train=False)
val_images = val_fmnist.data
val_targets = val_fmnist.targets

trn_dl, val_dl = get_data()
net, loss_fn, optimizer = get_model()

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
epochs = 5
for epoch in range(epochs):
    print(epoch)
    train_epoch_losses, train_epoch_accuracies = [], []
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, net, optimizer, loss_fn)
        train_epoch_losses.append(batch_loss)
    train_epoch_loss = np.array(train_epoch_losses).mean()

    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        is_correct = accuracy(x, y, net)
        train_epoch_accuracies.extend(is_correct)
    train_epoch_accuracy = np.mean(train_epoch_accuracies)

    for ix, batch in enumerate(iter(val_dl)):
        x, y = batch
        val_is_correct = accuracy(x, y, net)
        validation_loss = val_loss(x, y, net, loss_fn)
    val_epoch_accuracy = np.mean(val_is_correct)

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_losses.append(validation_loss)
    val_accuracies.append(val_epoch_accuracy)

epochs = np.arange(epochs) + 1

plt.subplot(211)
plt.plot(epochs, train_losses, 'bo', label='Training loss')
plt.plot(epochs, val_losses, 'r', label='Validation loss')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation loss with CNN')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.subplot(212)
plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation accuracy with CNN')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid('off')
plt.show()
