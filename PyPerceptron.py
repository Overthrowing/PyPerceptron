import numpy as np
from os.path import join
import time

# TODO: Clean imports
# import matplotlib.pyplot as plt

# from load_data import MnistDataloader


# Activation functions
def relu(x, derivative=True):
    if derivative:
        return np.where(x > 0, 1, 0)  # Derivative at 0 defined as 0
    else:
        return np.maximum(0, x)


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))


class Layer:
    def __init__(self, activation, size):
        self.activation = activation
        self.size = size

    def forward(self, input):
        """
        Pass input data throug the layer, dummy layer returns input
        Input:
        (input_size x batch_size) Matrix
        Output:
        (output_size x batch_size) Matrix
        """
        return input

    def backward(self, prev: "Layer", next: "Layer"):
        pass  # TODO


class InputLayer(Layer):
    def __init__(self, input: "np.array"):
        self.z = input

    def forward(self):
        pass


class Dense(Layer):
    def __init__(
        self,
        input_size,
        output_size,
        activation_func=sigmoid,
        learning_rate=0.1,
        initial_dist="normal",
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        match initial_dist:
            case "normal":
                self.weights = np.random.normal(
                    0, 1, size=(output_size, input_size)
                )  # TODO: Optimize
                self.biases = np.random.normal(0, 1, size=(output_size, 1))
            case "uniform":
                self.weights = np.random.uniform(-1, 1, size=(output_size, input_size))
                self.biases = np.random.uniform(-1, 1, size=(output_size, 1))

        self.z = np.array()
        self.activation = np.array()

    def forward(self, input: "np.array"):
        self.z = np.dot(self.weights, input.T) + self.bias
        self.activation = self.activation_func(self.z)

    def backward(self, prev: "Layer", next: "Layer"):
        self.error = np.dot(next.weights.T, next.error) * self.activation_func(
            self.z, derivative=True
        )
        weight_grad = (
            np.dot(self.error, prev.activation.T)
            / prev.activation.shape[1]  # Batch Size
        )
        self.weights -= weight_grad * self.learning_rate
        bias_grad = np.mean(self.error, axis=1, keepdims=True)
        self.biases -= bias_grad * self.learning_rate


class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self):
        for layer in self.layers:
            pass

    def train_batch(self, X, Y):
        pass

    def train(epochs, batch_size=100, learning_rate=0.1):
        pass
