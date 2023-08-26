import numpy as np
from os.path  import join

from load_data import MnistDataloader




# MNIST Dataset
input_path = 'Input/'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# Load MINST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist_dataloader.load_data()


# Function to convert raw value to array of 0s with a 1 at the label
def one_hot_encode(y, num_classes):
    one_hot = np.zeros((num_classes, 1))
    one_hot[y] = 1
    return one_hot


# Flatten Data
d1, d2, d3 = x_train_raw.shape
x_train = x_train_raw.reshape(d1, d2*d3)
y_train = np.array([one_hot_encode(y, 10) for y in y_train_raw], dtype=int)



d1, d2, d3 = x_test_raw.shape
x_test = x_test_raw.reshape(d1, d2*d3)




# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0) # Derivative at 0 defined as 0

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x)) # Can replace with np.multiply(x,y) if issues exist




# Input: 784x1, 2*Hidden: 16x1, Output: 10x1
# TODO: Test uniform as opposed to normal
hidden_1_w = np.random.normal(0, 1, size=(16, 784)) # hidden_1_w (16x784) * input (784x1) = hidden_1_a (16x1)
hidden_1_b = np.random.normal(0, 1, size=(16, 1)) # Bias for hidden layer 1
hidden_2_w = np.random.normal(0, 1, size=(16, 16)) # hidden_2_w (16x16) * hidden_1_a (16x1) = hidden_2_a (16x1)
hidden_2_b = np.random.normal(0, 1, size=(16, 1)) # Bias for hidden layer 2
output_w = np.random.normal(0, 1, size=(10, 16)) # output_w (10x16) * hidden_2_a (16x1) = output_a (10x1)
output_b = np.random.normal(0, 1, size=(10, 1)) # Bias for output layer, reccomended due to activation being sigmoid

epochs = 1000 # Hyperparameter
batch_size = 100
learning_rate = 0.1

for i in range(epochs):

    for j in range(0, x_train.shape[0], batch_size):
        # Forward
        hidden_1_z = np.dot(hidden_1_w, x_train[j:j+batch_size].T) + hidden_1_b # (16x100)
        hidden_1_a = sigmoid(hidden_1_z) # (16x100)
        hidden_2_z = np.dot(hidden_2_w, hidden_1_a) + hidden_2_b # (16x100)
        hidden_2_a = sigmoid(hidden_2_z) # (16x100)
        output_z = np.dot(output_w, hidden_2_a) + output_b # (10x100)
        output = sigmoid(output_z) # (10x100)

        cost = np.mean(np.square(output - np.squeeze(y_train[j:j+batch_size]).T))

        # Backwards

        # hidden_2_a - (16, 100)
        # output_z - (10, 100)
        # output (10, 100)

        # Output Layer
        # output_w_grad = np.dot((sigmoid_prime(output_z)*2*(output - np.squeeze(y_train[j:j+batch_size]).T)), hidden_2_a.T)
        # output_w -= output_w_grad*learning_rate
        # output_b_grad = sigmoid_prime(output_z)*2*(output - np.squeeze(y_train[j:j+batch_size]).T)
        # output_b -= np.mean(output_b_grad, axis=1, keepdims=True)*learning_rate

        output_error = sigmoid_prime(output_z)*2*(output - np.squeeze(y_train[j:j+batch_size]).T) # Remove multiplyer of 2 if issues exist? (10,100)
        output_w_grad = np.dot(output_error, hidden_2_a.T)/batch_size # Divide by batch size as each matrix contains data from batch_size inputs
        output_w -= output_w_grad * learning_rate
        output_b_grad = np.mean(output_error, axis=1, keepdims=True)
        output_b -= output_b_grad * learning_rate

        hidden_2_error = np.dot(output_w.T, output_error) * sigmoid_prime(hidden_2_z)
        hidden_2_w_grad = np.dot(hidden_2_error, hidden_1_a.T)/batch_size
        hidden_2_w -= hidden_2_w_grad * learning_rate
        hidden_2_b_grad = np.mean(hidden_2_error, axis=1, keepdims=True)
        hidden_2_b -= hidden_2_b_grad * learning_rate

        hidden_1_error = np.dot(hidden_2_w.T, hidden_2_error) * sigmoid_prime(hidden_1_z)
        hidden_1_w_grad = np.dot(hidden_1_error, x_train[j:j+batch_size])/batch_size
        hidden_1_w -= hidden_1_w_grad * learning_rate
        hidden_1_b_grad = np.mean(hidden_1_error, axis=1, keepdims=True)
        hidden_1_b -= hidden_1_b_grad * learning_rate

    print(f"Epoch: {i} | Cost: {cost}")


    