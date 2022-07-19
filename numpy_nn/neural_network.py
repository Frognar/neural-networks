import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from activation_functions import activation_function
from loss_functions import loss_function


def process(inputs, weights):
    pre_hidden = np.dot(inputs, weights[0]) + weights[1]
    hidden = activation_function(pre_hidden)
    return np.dot(hidden, weights[2]) + weights[3]


def feed_forward(inputs, outputs, weights):
    return loss_function(process(inputs, weights), outputs)


def update_weights(inputs, outputs, weights, lr):
    original_weights = deepcopy(weights)
    updated_weights = deepcopy(weights)
    original_loss = feed_forward(inputs, outputs, original_weights)
    for i, layer in enumerate(original_weights):
        for index, weight in np.ndenumerate(layer):
            temp_weights = deepcopy(weights)
            temp_weights[i][index] += 0.0001
            temp_loss = feed_forward(inputs, outputs, temp_weights)
            grad = (temp_loss - original_loss) / 0.0001
            updated_weights[i][index] -= grad*lr
    return updated_weights, original_loss


X = np.array([[1, 1], [0, 1], [0, 0], [1, 0]])
Y = np.array([[0], [1], [0], [1]])
W = [
    np.array([[-0.0053, 0.3793],
              [-0.5820, 0.3793],
              [-0.5820, -0.5204],
              [-0.2723, 0.1896],
              [-0.5820, 0.3793]], dtype=np.float32).T,
    np.array([-0.0140, 0.5607, -0.0628, 0.0620, -0.0600], dtype=np.float32),
    np.array([[0.1528, -0.1745, -0.1135, -0.1738, 0.1148]], dtype=np.float32).T,
    np.array([-0.5516], dtype=np.float32)
]

losses = []
for epoch in range(3000):
    W, loss = update_weights(X, Y, W, 0.1)
    losses.append(loss)

plt.plot(losses)
plt.title('Loss over increasing number of epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.show()

print(process(X, W))
