import numpy as np
import random as rd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def diff_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def backprop(self, x, y):
        # forming all the activations of each layers
        activation = x
        activations = [x]
        zs = []
        d_zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.reshape(np.dot(w, activation), (len(b), 1)) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            d_z = diff_sigmoid(z)
            d_zs.append(d_z)
        # the error for layer L (last layer)
        error_output = (activations[-1] - y) * d_zs[-1]
        errors = [error_output]

        # finding errors for rest of the layers (L-1, L-2, ... , 2, 1)
        L = self.num_layers
        for i in range(1, L - 1):
            tran_w = np.transpose(self.weights[-i])
            inner = [np.dot(w, errors[-1]) for w in tran_w]
            error = [inn * dz for inn, dz in zip(inner, d_zs[-i - 1])]
            errors.append(error)
        # errors.reverse()

        # the gradient of weights and biases
        grad_ws = []
        grad_bs = errors
        for i in range(L - 1):
            grad_w = [e * activations[i] for e in errors[i]]
            grad_ws.append(grad_w)
        # reshape the grad_ws matrix
        reshape_grad_ws = []
        for j in range(L - 1):
            reshape_grad_ws.append(
                np.vstack([np.transpose(gw) for gw in grad_ws[i]]))

        return (reshape_grad_ws, grad_bs)

    def update_mini_batch(self, mini_batch, eta):
        for y, x in mini_batch:
            d_w, d_b = self.backprop(x, y)
            self.weights = [w - (eta/len(mini_batch)) *
                            dw for w, dw in zip(self.weights, d_w)]
            self.biases = [b - (eta/len(mini_batch)) *
                           db for b, db in zip(self.biases, d_b)]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
            return a

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for x, y in test_data]
        return sum([int(x == y) for x, y in test_results])

    def SGD(self, training_data, mini_batch_size, epoches, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n_train = len(training_data)

        for i in range(epoches):
            rd.shuffle(training_data)
            mini_batches = [training_data[j: j + mini_batch_size]
                            for j in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print('Epoch {0}: {1} / {2}'.format(i,
                                                    self.evaluate(test_data), n_test))
            else:
                print('Epoch {0} complete'.format(i))


# net = Network([2, 3, 4, 1])
# input_act = np.array([[1], [0]])
# desired_output = np.array([1])
# print(net.weights)
# print('------------------------------')
# net.backprop(input_act, desired_output)
