import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


def relu(x):
    return np.where(x > 0, x, 0)


def linear(x):
    return x


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


def activation_function(x):
    return sigmoid(x)
