import numpy as np

from .utils import sigmoid


class Layer:
    """The base class for all layers, initializes the layer to zeros

    :param size: number of neurons in the layer (output size)
    :type size: int
    :param activation_func: activation function (with derivative as a boolean parameter) for the layer
    :type activation_func: function, optional
    """

    def __init__(self, size, activation_func=sigmoid):
        self.size = size
        self.activation_func = activation_func
        self.z = np.zeros((size, 1))  # Non-activated neuron values
        self.a = np.zeros(
            (size, 1)
        )  # Activated neuron values (activation_func applied to z)

    def forward(self):
        """Forward pass of the layer, by default does nothing

        TODO:: implement default forward call such that it returns activation
        """
        pass

    def backward(self):
        """Backward pass of the layer, by default does nothing"""
        pass


class Dense(Layer):
    """Layer that is fully connected to previous layer

    :param size: number of neurons in the layer
    :type size: int
    :param activation: activation function (with derivative as a boolean parameter) for the layer
    :type activation: function
    :param learning_rate: learning rate for the layer, defaults to 0.1
    :type learning_rate: float, optional
    """

    def __init__(
        self, size, activation_func=sigmoid, learning_rate=0.1, initial_dist="normal"
    ):
        super().__init__(size, activation_func)
        self.learning_rate = learning_rate

        self.w = None
        self.b = None
        self.initial_dist = initial_dist

    def initialize_weights(self, prev_size):
        """Initializes the weights and biases of the layer

        :param prev_size: number of neurons in the previous layer
        :type prev_size: int
        :param initial_dist: distribution to initialize weights with, defaults to a normal distribution
        :type initial_dist: str, optional
        """
        match self.initial_dist:
            case "normal":
                self.w = np.random.normal(0, 1, size=(self.size, prev_size))
                self.b = np.random.normal(0, 1, size=(self.size, 1))
            case "uniform":
                self.w = np.random.uniform(-1, 1, size=(self.size, prev_size))
                self.b = np.random.uniform(-1, 1, size=(self.size, 1))

    def forward(self, prev: Layer):
        """Forward pass of the layer

        :param prev: previous layer
        :type prev: Layer
        """
        self.z = np.dot(self.w, prev.a) + self.b
        self.a = self.activation_func(self.z)

    def backward(self, prev: Layer, next: Layer = None, y: "np.array" = None):
        """Backward pass of the layer

        :param prev: previous layer
        :type prev: Layer
        :param next: next layer
        :type next: Layer, optional
        :param y: labels, defaults to None
        :type y: numpy array, optional
        """
        if isinstance(y, np.ndarray):
            self.error = self.activation_func(self.z, derivative=True) * (self.a - y)
        else:
            self.error = np.dot(next.w.T, next.error) * self.activation_func(
                self.z, derivative=True
            )
        batch_size = prev.a.shape[1]
        weight_grad = np.dot(self.error, prev.a.T) / batch_size
        self.w -= weight_grad * self.learning_rate
        bias_grad = np.mean(self.error, axis=1, keepdims=True)
        self.b -= bias_grad * self.learning_rate
