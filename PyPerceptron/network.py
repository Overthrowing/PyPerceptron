import numpy as np

from .layers import Layer
from .utils import one_hot_encode


class Network:
    """The perceptron model"""

    def __init__(self):
        self.layers = []
        self.cost = 1

    def add_layer(self, layer):
        """Adds a layer to the network

        :param layer: layer to be added
        :type layer: Layer
        """
        # Initialize weights and biases if not input layer
        if len(self.layers) > 0:
            layer.initialize_weights(self.layers[-1].size)

        self.layers.append(layer)

    def forward(self, x):
        """Forward pass of the network

        :param x: input to the network
        :type x: numpy.array

        :raises ValueError: if first layer is a connected layer such as a Dense layer

        :return: output of the network (activations of the output layer)
        :rtype: numpy.array

        """
        if not isinstance(self.layers[0], Layer):
            raise ValueError(
                "First layer of the network must have no connections, use Layer instead of Dense or other connected layer"
            )
        self.layers[0].a = x
        for layer_idx in range(1, len(self.layers)):
            prev_layer = self.layers[layer_idx - 1]
            self.layers[layer_idx].forward(prev_layer)

        return self.layers[-1].a

    def back_prop(self, y):
        """Performs back propagation accross all layers of the network

        :param y: labels (one hot encoded)
        :type y: numpy.array
        """
        self.layers[-1].backward(self.layers[-2], y=y)

        # Backpropagate through all layers except the first and last
        for layer_idx in range(len(self.layers) - 2, 0, -1):
            prev_layer = self.layers[layer_idx - 1]
            cur_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]

            cur_layer.backward(prev_layer, next_layer)

    def train_batch(self, x, y):
        """Trains the network on a batch of data

        :param x: input data
        :type x: numpy.array
        :param y: labels (one hot encoded)
        :type y: numpy.array
        """
        self.forward(x)
        self.back_prop(y)
        self.cost = np.mean(np.square(self.layers[-1].a - y))

    def train(self, x_data, y_labels, epochs, batch_size):
        """Trains the network on the given data

        :param x_data: input data
        :type x_data: numpy.array
        :param y_labels: labels
        :type y_labels: numpy.array
        :param epochs: number of epochs to train for
        :type epochs: int
        :param batch_size: size of each batch
        :type batch_size: int
        :param learning_rate: learning rate, defaults to 0.1
        :type learning_rate: float, optional

        ::TODO: add learning rate as a parameter
        """
        # Convert to column vectors
        x = x_data.T
        y = one_hot_encode(y_labels).T
        for epoch in range(epochs):
            for batch_idx in range(0, x_data.shape[0], batch_size):
                # Get batch of data in the form of column vectors
                x_batch = x[:, batch_idx : batch_idx + batch_size]
                y_batch = y[:, batch_idx : batch_idx + batch_size]

                self.train_batch(x_batch, y_batch)
            print(f"Cost: {self.cost} | Epoch {epoch}/{epochs}")

    def predict(self, x):
        """Predicts the output of the network for the given input

        :param x: input data
        :type x: numpy.array
        :return: output of the network
        :rtype: numpy.array
        """
        x = np.reshape(x, [x.shape[-1], 1])
        return np.argmax(self.forward(x))
