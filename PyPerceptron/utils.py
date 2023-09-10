import numpy as np


# Activation functions with derivative
def relu(x, derivative=False):
    """Rectified Linear Unit activation function with derivative

    :param x: input to the activation function
    :type x: float
    :param derivative: whether to return the derivative of the function, defaults to False
    :type derivative: bool, optional
    :return: output/derivative of the activation function at x
    :rtype: float
    """
    if derivative:
        return np.where(x > 0, 1, 0)  # Derivative at 0 defined as 0
    else:
        return np.maximum(0, x)


def sigmoid(x, derivative=False):
    """Sigmoid activation function with derivative

    :param x: input to the activation function
    :type x: float
    :param derivative: whether to return the derivative of the function, defaults to False
    :type derivative: bool, optional
    :return: output/derivative of the activation function at x
    :rtype: float
    """
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))


def one_hot_encode(y, num_classes: int = None):
    """Converts array of labels to output layer like column vectors of 0s with 1 at the index of the correct label

    :param y: array of labels
    :type y: numpy array
    :param num_classes: number of classes in the dataset
    :type num_classes: int, optional
    :return: one hot encoded array
    :rtype: numpy array
    """
    if not num_classes:
        num_classes = np.max(y) + 1
    return np.eye(num_classes)[y]
