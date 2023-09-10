# PyPerceptron

A neural network (multilayer perceptron) implementation from scratch using Numpy

## Inspiration
This project was inspired by the incredible [video series by 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&pp=iAQB) covering neural networks from the ground up

## Setup

(PIP)

```shell
python -m pip install
```

(Poetry)

```shell
poetry install
```

## Quickstart
Using the [MNIST Handwritten Digits](examples/MNIST%20Handwritten%20Digits/recognize_digits.py) example

### Importing the library
```python
from PyPerceptron import Network, layers
```

### Training Data
See [Eexample](examples/load_data.py) for more details

```python
# Load the MNIST data
...

# Normalized Data
x_train # 60000 flattended 28x28 pixel images (60000, 784)
y_train # 60000 labels (60000, 1)

x_test # 10000 flattended 28x28 pixel images (10000, 784)
y_test # 10000 labels (10000, 1)
```

### Creating the Network
```python
model = Network()

model.add_layer(layers.Layer(784)) # input layer with 784 neurons representing 784 pixels, not using dense layer as input layer has no connections
model.add_layer(layers.Dense(16)) # 16 neuron hidden layer
model.add_layer(layers.Dense(16)) # 16 neuron hidden layer
model.add_layer(layers.Dense(10)) # output layer with 10 neurons representing 10 classes
```

### Training the Network
Feel free to experiment with the hyperparameters and observe their effects on the model's performance
```python
model.train(x_train, y_train, epochs=100, batch_size=50)
```

### Evaluating the Network
Testing the model's accuracy on the test dataset and printing incorrect predictions for manual inspection
```python
data = x_test
labels = y_test

c = 0
for i in range(len(data)):
    if (p := model.predict(data[i])) != (l := labels[i]):
        c += 1
        print(f"{i} - P: {p} | A: {l}")
print(f"Accuracy: {1-c/len(data)}")
```

## Layers

### Layer
The base layer class, does not have connections, weights, or biases. Can be used for input layers
 
```python
model.add_layer(layers.Layer(784)) # Input layer with 784 neurons
```

### Dense
A fully connected layer that should be used for hidden and output layers

```python
model.add_layer(layers.Dense(16, activation_func=sigmoid, learning_rate=0.1, initial_dist="normal")) # Hidden layer with 16 neurons, sigmoid activation functions, 0.1 learning rate, and normal distribution for initial weights and biases
```
#### Initial Distribution
The initial distribution of the weights and biases can be set to either "normal" or "uniform" (default is "uniform")

#### Activation Functions
The activation functions can be set to either sigmoid or relu (default is sigmoid) and can be imported from the utils module

```python
from PyPerceptron import utils

sigmoid = utils.sigmoid
relu = utils.relu
```

## Explanation
### Structure
Each layer of the network is composed of a column vector containing it's activation (neuron values) and a weight matrix of size (cur_layer_size x prev_layer_size) containing the weights of the connections between the current layer and the previous layer. The bias vector is also stored in each layer and is added to the activation vector after the weights are applied. The activation function is applied to the activation vector after the bias is added in order to introduce non-linearity to the network.

### Training
Training consists of taking a batch of training data and training labels and passing them through the network using a forward pass and back propagation step. The forward pass consists of passing the training data through the network and storing the activations of each layer. The back propagation step consists of calculating the error of the output layer and using that to calculate the error of each previous layer. The error of each layer is then used to calculate the partial derivatives of the loss function with respect to the weights and biases of each layer. These partial derivatives are averaged across each batch to so that we make one overall change for each batch as opposed to a change for every training example. The weights and biases are then updated using these partial derivatives (gradient) and are scaled by the learning rate which dictates how much the weights and biases may change at each step of training. This process is repeated for each batch of training data for a specified number of epochs.

### Prediction
Prediction consists of passing the test data through the network and returning the index of the neuron with the highest activation as the predicted label. The predicted label is then compared to the actual label and the accuracy is calculated.


#### Resources I Found Helpful:
- [Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&pp=iAQB) - 3Blue1Brown
- _Neural Networks: A Systematic Introduction_ - Raúl Rojas
- Phylliida (https://stats.stackexchange.com/users/78563/phylliida), Backpropagation algorithm and error in hidden layer, URL (version: 2021-07-26): https://stats.stackexchange.com/q/155658




