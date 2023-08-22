import numpy as np
from os.path  import join

from load_data import MnistDataloader


# Activation functions
def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return np.where(x > 0, 1, 0) # Derivative at 0 defined as 0

# MNIST Dataset
input_path = 'Input/'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# Load MINST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train_raw, y_train), (x_test_raw, y_test) = mnist_dataloader.load_data()

# Flatten Data
d1, d2, d3 = x_train_raw.shape
x_train = x_train_raw.reshape(d1, d2*d3)

d1, d2, d3 = x_test_raw.shape
x_test = x_test_raw.reshape(d1, d2*d3)



# Input: 728x1, 2*Hidden: 16x1, Output: 10x1