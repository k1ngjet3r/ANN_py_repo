import random
from math import exp
import numpy as np


# class Network:
#     def __init__(self, sizes):
#         self.num_layers = len(sizes)
#         self.sizes = sizes
#         self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
#         self.weights = [np.random.randn(y, x)
#                         for x, y in zip(sizes[:-1], sizes[1:])]


# net = Network([2, 3, 1])

# # print(net.biases)
# # print('------------------------')
# nabla_b = [np.zeros(b.shape) for b in net.biases]


# print(net.biases)
# print('------------------------')
# # print(net.weights)

# # print('------------------------')
# # print(nabla_b)
# list(zip(net.biases, net.weights))


# weight = [['a', 'b'], ['d', 'e'], ['g', 'h']]
# biases = ['1', '2']

# # for b, w in zip(net.biases, net.weights):
# #    print(b, w)
# #    print('-----------------------')

# print(nabla_b)

# print(np.transpose(weight))


a = [1, 2, 3]

a[-1] = 4

print(a)
