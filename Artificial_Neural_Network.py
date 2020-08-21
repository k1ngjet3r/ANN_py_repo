import numpy as np
from math import exp
import tensorflow as tf
from random import shuffle

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

fraction = 0.99/255.0
x_train = x_train * fraction + 0.01
x_test = x_test * fraction + 0.01

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


def data_formatter(x, y):
    return [(xx, yy) for xx, yy in zip(x, y)]


training_data = data_formatter(x_train, desired_y_train)
testing_data = data_formatter(x_test, desired_y_test)


def activation_function(z):
    return 1 / (1 + exp(-z))


def diff_activation_function(z):
    return activation_function(z) * (1 - activation_function(z))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1).flatten() for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def backprop(self, input_layer, desired_output):
        # setting the shape of the nabla_b and nabla_w
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Setting up the required list
        activation = input_layer.flatten()
        activations = [input_layer.flatten()]
        zs = []

        # Feeding forward and we got the full picture of the network including every neurons (a) and weighted sum (z) in each layers
        for i in range(self.num_layers - 1):
            weighted_sum = np.add(
                np.dot(self.weights[i], activation), self.biases[i].flatten())
            zs.append(weighted_sum)
            activation = np.array([activation_function(z)
                                   for z in weighted_sum])
            activations.append(activation)

        # Backward pass, last layer first
        delta = (activations[-1] - desired_output) * \
            np.array([diff_activation_function(z) for z in zs[-1]])
        nabla_b[-1] = delta
        nabla_w[-1] = np.array([(i * activations[-2].transpose()).tolist()
                                for i in delta])

        # Other layers
        for j in range(2, self.num_layers):
            z = zs[-j]
            activation_prime = np.array(
                [diff_activation_function(zz) for zz in z])
            delta = np.dot(
                self.weights[-j + 1].transpose(), delta) * activation_prime
            nabla_b[-j] = delta
            nabla_w[-j] = np.array([(i * activations[-j -
                                                     1].transpose()).tolist() for i in delta])

        return (nabla_b, nabla_w)

    def update_weights_and_biases(self, nabla_w, nabla_b, learning_rate, data_mini_batch_size):
        self.weights = [w - (learning_rate/data_mini_batch_size)
                        * n_w for w, n_w in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate/data_mini_batch_size)
                       * n_b for b, n_b in zip(self.biases, nabla_b)]
        # return self.weights, self.biases

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def feed_forward(self, a):
        a = a.flatten()
        for b, w in zip(self.biases, self.weights):
            weighted_sum = np.add(np.dot(w, a), b)
            a = np.array([activation_function(z) for z in weighted_sum])
        return a

    def SGD(self, training_data, batch_size, epoch, learning_rate, test_data=None):
        if test_data != None:
            n_test = len(test_data)
        n = len(training_data)
        for i in range(epoch):
            print('running epoch no. {0}'.format(i + 1))
            shuffle(training_data)
            mini_batches = [training_data[k: k + batch_size]
                            for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                for x, y in mini_batch:
                    nabla_b, nabla_w = self.backprop(x, y)
                    self.update_weights_and_biases(
                        nabla_w, nabla_b, learning_rate, batch_size)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(i,
                                                    self.evaluate(test_data), n_test))
                print('Accuracy: {0}%'.format(
                    (self.evaluate(test_data) / n_test)*100))
            else:
                print('Epoch {0} complete'.format(i))


net = Network([784, 30, 10])
net.SGD(training_data, batch_size=10, epoch=30,
        learning_rate=3.0, test_data=testing_data)
