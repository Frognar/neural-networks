import numpy as np


def mse(p, y):
    return np.mean(np.square(p - y))


def mae(p, y):
    return np.mean(np.abs(p - y))


def binary_cross_entropy(p, y):
    return -np.mean(np.sum((y*np.log(p) + (1-y)*np.log(1-p))))


def categorical_cross_entropy(p, y):
    return -np.mean(np.sum(y*np.log(p)))


def loss_function(p, y):
    return mse(p, y)
