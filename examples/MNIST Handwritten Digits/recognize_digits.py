from os.path import join

from PyPerceptron import Network, layers
from load_data import MnistDataloader


# MNIST Dataset
input_path = "Input/"
training_images_filepath = join(
    input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"
)
training_labels_filepath = join(
    input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
)
test_images_filepath = join(input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
test_labels_filepath = join(input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

# Load MINST dataset
mnist_dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
)
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist_dataloader.load_data()

# Flatten Data
d1, d2, d3 = x_train_raw.shape
x_train = x_train_raw.reshape(d1, d2 * d3)
y_train = y_train_raw

d1, d2, d3 = x_test_raw.shape
x_test = x_test_raw.reshape(d1, d2 * d3)
y_test = y_test_raw


# Create Network
model = Network()

model.add_layer(
    layers.Layer(784)
)  # input layer with 784 neurons representing 784 pixels, not using dense layer as input layer has no connections
model.add_layer(layers.Dense(16))
model.add_layer(layers.Dense(16))
model.add_layer(
    layers.Dense(10)
)  # output layer with 10 neurons representing 10 classes


# Train Network
model.train(x_train, y_train, epochs=100, batch_size=50)

data = x_test
labels = y_test

# Test Accuracy and Print Incorrect Predictions
c = 0
for i in range(len(data)):
    if (p := model.predict(data[i])) != (l := labels[i]):
        c += 1
        print(f"{i} - P: {p} | A: {l}")
print(f"Accuracy: {1-c/len(data)}")
