import random
from math import exp
import numpy as np


def Sigmoid(q):
    return 1 / (1 + exp(-q))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def backprop(self, x, y):
        nebla_b = [np.zeros(b.shape) for b in self.biases]
        zs = []
        act = x
        acts = [x]

        # feed forward
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, act) + b
            zs.append(z)
            act = Sigmoid(z)
            acts.append(act)
