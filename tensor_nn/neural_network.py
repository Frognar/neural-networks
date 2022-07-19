import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt


def process(inputs, weights, activation):
    pre_hidden = inputs @ weights[0] + weights[1]
    hidden = activation(pre_hidden)
    return hidden @ weights[2] + weights[3]


X = torch.tensor([[1., 1.], [0., 1.], [0., 0.], [1., 0.]])
Y = torch.tensor([[0.], [1.], [0.], [1.]])
W = [
    torch.rand((2, 5), requires_grad=True),
    torch.rand((5,), requires_grad=True),
    torch.rand((5, 1), requires_grad=True),
    torch.rand((1,), requires_grad=True)
]
opt = SGD(W, lr=0.1)
loss_function = nn.MSELoss()
activation_function = nn.Sigmoid()

losses = []
for epoch in range(3000):
    opt.zero_grad()
    pred_Y = process(X, W, activation_function)
    loss = loss_function(pred_Y, Y)
    loss.backward()
    opt.step()
    losses.append(loss.item())

plt.plot(losses)
plt.title('Loss over increasing number of epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.show()

print(process(X, W, activation_function))
