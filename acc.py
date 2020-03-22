import random
from math import exp
import numpy as np


def Sigmoid(x):
    return 1 / (1 + exp(-q))


def diff_sigmoid(x):
    return Sigmoid(x) * (1 - Sigmoid(x))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def backprop(self, x, y):
        zs = []
        act = x
        acts = [x]
        # feed forward
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, act) + b
            zs.append(z)  # zs = [z1, z2, z3, ..., zL]
            act = [Sigmoid(zz) for zz in z]
            acts.append(act)  # acts = [a0, a1, a2, ..., aL]

        # back propagate
        # last hidden layer
        deltas = []
        cost_diff_a = acts[-1] - y
        Delta = np.dot(cost_diff_a, diff_sigmoid(zs[-1]))
        deltas.append(Delta)  # deltas = [dL]

        # other hidden layers
        for j in range(1, self.num_layers):
            inner_prod = np.dot(self.weights[-j].transpose(), deltas[-1])
            delta = np.dot(inner_prod, diff_sigmoid(zs[-j - 1]))
            deltas.append(delta)  # equals to nebla_b

        # nebla_w
        nebla_w = [
            np.dot(acts[-k - 1], deltas[k - 1]) for k in range(1, self.num_layers)
        ]

        # nebla_b
        nebla_b = deltas

        return (nebla_w, nebla_b)

    def update_mini_datch(self, mini_batch, learning_rate):
        for x, y in mini_batch:
            nebla_w, nebla_w = self.backprop(x, y)
            self.weights = [
                w - (learning_rate / len(mini_batch)) * nw
                for w, nw in zip(self.weights, nebla_w)
            ]
            self.biases = [
                b - (learning_rate / len(mini_batch)) * nb
                for b, nb in zip(self.biases, nebla_b)
            ]

