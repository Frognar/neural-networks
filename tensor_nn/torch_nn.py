import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt


class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(2, 8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8, 1)

    def forward(self, inputs):
        out = self.input_to_hidden_layer(inputs)
        out = self.hidden_layer_activation(out)
        out = self.hidden_to_output_layer(out)
        return out


class MyDataset(Dataset):
    def __init__(self, inputs, target):
        self.x = torch.tensor(inputs).float()
        self.y = torch.tensor(target).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


X = [[1, 2], [3, 4], [5, 6], [7, 8], [20, 0], [0, 15], [0, 0], [12.9, 3.1]]
Y = [[3], [7], [11], [15], [20], [15], [0], [16]]
dataset = MyDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
net = MyNeuralNet()

if os.path.exists('model.pth'):
    state_dict = torch.load('model.pth')
    if state_dict is not None:
        net.load_state_dict(state_dict)

loss_func = nn.MSELoss()
opt = SGD(net.parameters(), lr=0.001)

loss_history = []
for _ in range(1000):
    for x, y in dataloader:
        opt.zero_grad()
        pred_y = net(x)
        loss = loss_func(pred_y, y)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())

torch.save(net.state_dict(), 'model.pth')

plt.plot(loss_history)
plt.title('Loss variation over increasing epochs')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.show()

val_x = torch.tensor([[8, 9], [10, 11], [1.5, 2.5], [19.9, .1], [99.9, .1]])
print(net(val_x))
