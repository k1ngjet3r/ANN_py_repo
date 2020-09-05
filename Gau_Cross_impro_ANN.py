# improvement_elements

import numpy as np
from math import exp


def sigmoid(z):
    return 1 / (1 + exp(z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network:
    def __init__(self, sizes, cost_function):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # initialized the weights using Gaussian distribution
        self.weights = [
            np.random.randn(y, x) / np.squrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]
        self.cost_function = cost_function

    # Defining the cross_entropy cost function
    def cross_entropy_cost_function(self, desired_outputs, outputs, n):
        natural_log_a = np.asarray([np.log(a) for a in outputs])
        natural_log_1_a = np.asarray([np.log(1 - a) for a in outputs])
        ce_function = [
            -(1 / n) * (y * ln_a + (1 - y) * ln_1_a)
            for y, ln_a, ln_1_a in zip(desired_outputs, natural_log_a, natural_log_1_a)
        ]
        return ce_function

