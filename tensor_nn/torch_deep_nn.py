from torchvision import datasets
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch import optim
from torch.optim import Adam
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_folder = '~/data/FMNIST'
fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
tr_images = fmnist.data
tr_targets = fmnist.targets
val_fmnist = datasets.FashionMNIST(data_folder, download=True, train=False)
val_images = val_fmnist.data
val_targets = val_fmnist.targets


class FMNISTDataset(Dataset):
    def __init__(self, inputs, targets):
        inputs = inputs.float()
        inputs = inputs.view(-1, 28 * 28) / 255
        self.x, self.y = inputs, targets

    def __getitem__(self, index):
        return self.x[index].to(device), self.y[index].to(device)

    def __len__(self):
        return len(self.x)


def get_model():
    model = nn.Sequential(
        nn.Linear(28 * 28, 200),
        nn.Dropout(.2),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.Dropout(.2),
        nn.ReLU(),
        nn.Linear(200, 10)
    ).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_function, optimizer


def train_batch(inputs, targets, model, optimizer, loss_function):
    model.train()
    optimizer.zero_grad()
    prediction = model(inputs)
    l2_regularization = 0
    for param in net.parameters():
        l2_regularization += torch.norm(param, 2)
    loss = loss_function(prediction, targets) + 0.01 * l2_regularization
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def accuracy(inputs, targets, model):
    model.eval()
    prediction = model(inputs)
    max_values, argmaxes = prediction.max(-1)
    correct = argmaxes == targets
    return correct.cpu().numpy().tolist()


def get_data():
    train = FMNISTDataset(tr_images, tr_targets)
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    validation = FMNISTDataset(val_images, val_targets)
    validation_dl = DataLoader(validation, batch_size=len(val_images), shuffle=False)
    return train_dl, validation_dl


@torch.no_grad()
def val_loss(inputs, targets, model):
    prediction = model(inputs)
    loss = loss_fn(prediction, targets)
    return loss.item()


trn_dl, val_dl = get_data()
net, loss_fn, opt = get_model()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    factor=0.5,
    patience=0,
    threshold=0.001,
    verbose=True,
    min_lr=1e-5,
    threshold_mode='abs'
)
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
epochs = 30
for epoch in range(epochs):
    print(epoch)
    train_epoch_losses, train_epoch_accuracies = [], []
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, net, opt, loss_fn)
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
        validation_loss = val_loss(x, y, net)
        scheduler.step(validation_loss)
    val_epoch_accuracy = np.mean(val_is_correct)

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_losses.append(validation_loss)
    val_accuracies.append(val_epoch_accuracy)


epochs = np.arange(epochs)+1
plt.subplot(211)
plt.plot(epochs, train_losses, 'bo', label='Training loss')
plt.plot(epochs, val_losses, 'r', label='Validation loss')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation loss with learning rate scheduler')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.subplot(212)
plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation accuracy with learning rate scheduler')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()
plt.grid('off')
plt.show()
