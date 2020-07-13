import numpy as np
from math import exp
import tensorflow as tf
from random import shuffle

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
fraction = 0.99/255.0
x_train, x_test = (np.around(x_train * fraction, decimals=2) +
                   0.01), (np.around(x_test * fraction, decimals=2) + 0.01)

desired_y_train = []
desired_y_test = []

for i in y_train:
    zero = np.zeros(10)
    zero[i] = 1
    desired_y_train.append(zero)

for j in y_test:
    zero = np.zeros(10)
    zero[j] = 1
    desired_y_test.append(zero)


def data_formater(x, y):
    return [(xx, yy) for xx, yy in zip(x, y)]


training_data = data_formater(x_train, y_train)
testing_data = data_formater(x_test, y_test)


def Sigmoid(z):
    return 1 / (1 + exp(-z))


def diff_Sigmoid(z):
    return Sigmoid(z) * (1 - Sigmoid(z))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.around(np.random.randn(y, 1), decimals=2)
                       for y in sizes[1:]]
        self.weights = [np.around(np.random.randn(y, x), decimals=2)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # Function that calculate and forming the output layer neurons from inputs
    def feed_forward(self, input_layer):
        activations = [input_layer.flatten()]
        activation = np.array(input_layer.flatten())
        zs = []
        for i in range(self.num_layers - 1):
            weighted_sum = np.add(
                np.dot(self.weights[i], activation), self.biases[i].flatten())
            zs.append(weighted_sum)
            activation = np.array([Sigmoid(z) for z in weighted_sum])
            activations.append(activation)
        return activations, zs

    def error_generator(self, ann_network, zs):
        errors = []
        d_sigmoid_z = [np.array([diff_Sigmoid(zij) for zij in z]) for z in zs]
        gradient_L = np.dot(self.weights[-1].transpose(), ann_network[-1])
        backward_errors = [
            np.array([g * z for g, z in zip(gradient_L, d_sigmoid_z[-1])])]
        for i in range(1, self.num_layers - 1):
            gradient_l = np.dot(
                self.weights[-i].transpose(), backward_errors[-1])
            error_l = np.array(
                [g * z for g, z in zip(gradient_l, d_sigmoid_z[-i - 1])])
            backward_errors.append(error_l)

        for j in range(len(backward_errors)):
            errors.append(backward_errors[-1 - j])

        nabla_b = [np.array([[j] for j in i]) for i in errors]
        nabla_w = [np.array([(ann_network[i] * d).tolist()
                             for d in errors[i]]) for i in range(net.num_layers - 1)]

        return nabla_b, nabla_w

    def update_weights_and_biases(self, nabla_w, nabla_b, learning_rate, data_mini_batch_size):
        self.weights = [w - (learning_rate/data_mini_batch_size)
                        * n_w for w, n_w in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate/data_mini_batch_size)
                       * n_b for b, n_b in zip(self.biases, nabla_b)]
        # return self.weights, self.biases

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)[0][-1]), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(self, training_data, batch_size, epoch, learning_rate, test_data=None):
        if test_data != None:
            n_test = len(test_data)
        n = len(training_data)
        for i in range(epoch):
            shuffle(training_data)
            mini_batches = [training_data[k: k + batch_size]
                            for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                for x, y in mini_batch:
                    activations, zs = self.feed_forward(x)
                    nabla_b, nabla_w = self.error_generator(activations, zs)
                    self.update_weights_and_biases(
                        nabla_w, nabla_b, learning_rate, batch_size)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(i,
                                                    self.evaluate(test_data), n_test))
            else:
                print('Epoch {0} complete'.format(i))


net = Network([784, 30, 10])
net.SGD(training_data, 10, 10, 3.0, test_data=testing_data)
