import numpy as np
import json
import os

# --- Neural Network Class --- #
class Layer:
    def __init__(self, size, input_size, activation_function):
        self.size = size
        self.input_size = input_size
        self.activation_function = activation_function
        self.activation_derivative = None
        if activation_function == relu:
            self.activation_derivative = relu_derivative
        elif activation_function == sigmoid:
            self.activation_derivative = sigmoid_derivative

        self.weights = np.random.randn(input_size, size) * 0.01
        self.biases = np.zeros((1, size))
        self.z = None
        self.a = None

        self.da = None # Gradient of the loss with respect to the activation of the layer
        self.dz = None # Gradient of the loss with respect to the z of the layer
        self.dw = None # Gradient of the loss with respect to the weights
        self.db = None # Gradient of the loss with respect to the biases

class Network:
    def __init__(self):
        self.layers = []

    def addLayer(self, size, input_size, activation_function):
        layer = Layer(size, input_size, activation_function)
        self.layers.append(layer)

    def removeLayer(self, index):
        if 0 <= index < len(self.layers):
            del self.layers[index]
        else:
            raise IndexError("Layer index out of range")
        
    def forward(self, inputs): # input shape (1, input_size)
        if inputs.shape[1] != self.layers[0].input_size:
            raise ValueError(f"Input size {inputs.shape[1]} does not match the expected size {self.layers[0].input_size}")
        x = inputs

        for i in range(len(self.layers)):
            layer = self.layers[i]
            z = np.dot(x, layer.weights) + layer.biases
            a = layer.activation_function(z)
            layer.z = z
            layer.a = a
            x = a
        return a
    
    def backward(self, y_true, y_pred, X, learning_rate=0.01): # X shape (batch_size, input_size); y_true shape (batch_size, output_size)
        batch_size = y_true.shape[0]
        error = y_pred - y_true
        loss = cross_entropy_loss(y_true, y_pred)
        last_layer = self.layers[-1]

        last_layer.dz = error
        last_layer.dw = np.dot(self.layers[-2].a.T, last_layer.dz) / batch_size
        last_layer.db = np.sum(last_layer.dz, axis=0, keepdims=True) / batch_size

        last_layer.weights -= learning_rate * last_layer.dw
        last_layer.biases -= learning_rate * last_layer.db

        for i in reversed(range(len(self.layers) -1)):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            prev_a = self.layers[i - 1].a if i > 0 else X

            # calculate gradients
            layer.da = np.dot(next_layer.dz, next_layer.weights.T)
            layer.dz = layer.da * layer.activation_derivative(layer.a)
            layer.dw = np.dot(prev_a.T, layer.dz) / batch_size
            layer.db = np.sum(layer.dz, axis=0, keepdims=True) / batch_size

            # update weights and biases
            layer.weights -= learning_rate * layer.dw
            layer.biases -= learning_rate * layer.db
        return loss
    
# --- Activation functions --- #
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def raw(x):
    return x
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# --- Loss functions --- #
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-15)) / m
    return loss

net = Network()

def create_network(input_size, output_size, hidden_layers): 
    # hiddenlayers: [size, activation_function, amount]
    
    for segment in hidden_layers: # hidden layers
        if segment[1] == "sigmoid":
            activation_function = sigmoid
        elif segment[1] == "relu":
            activation_function = relu
        for i in range(segment[2]):
            if i == 0:
                net.addLayer(segment[0], input_size, activation_function)
            else:
                net.addLayer(segment[0], net.layers[-1].size, activation_function)
    
    net.addLayer(output_size, net.layers[-1].size, softmax) # Output layer

def get_network():
    return net
def clear_network():
    net.layers = []

def train(X, y, epochs, batch_size, learning_rate, save_interval, filename=None):
    """
    Train the network using mini-batch gradient descent.\n\n
    :param X: Input data (number of samples, inputs) (rows: samples, columns: inputs)\n
    :param y: Target data (number of samples, outputs) (rows: samples, columns: outputs)
    """
    num_samples = X.shape[0]
    for epoch in range(epochs):
        for i in range(0, num_samples, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Forward pass
            y_pred = net.forward(X_batch)

            # Backward pass
            loss = net.backward(y_batch, y_pred, X_batch, learning_rate)
        print(f"Epoch {epoch + 1}/{epochs}, Batch {i // batch_size + 1}, Loss: {loss:.4f}")
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            save_network(filename)
            print(f"Network saved")
def test(X, y):
    """
    Test the network on the test data.\n\n
    :param X: Input data (number of samples, inputs) (rows: samples, columns: inputs)\n
    :param y: Target data (number of samples, outputs) (rows: samples, columns: outputs)
    """
        
    predictions = net.forward(X)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
    # print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy, predictions

def save_network(filename):
    network_data = {
        'layers': []
    }
    for layer in net.layers:
        layer_data = {
            'size': layer.size,
            'input_size': layer.input_size,
            'activation_function': layer.activation_function.__name__,
            'weights': layer.weights.tolist(),
            'biases': layer.biases.tolist()
        }
        network_data['layers'].append(layer_data)
    with open(filename, "w") as f:
        json.dump(network_data, f)

def load_network(filename):
    with open(filename, "r") as f:
        network_data = json.load(f)
    
    for layer_data in network_data["layers"]:
        if layer_data["activation_function"] == "sigmoid":
            activation_function = sigmoid
        elif layer_data["activation_function"] == "relu":
            activation_function = relu
        elif layer_data["activation_function"] == "softmax":
            activation_function = softmax
        elif layer_data["activation_function"] == "raw":
            activation_function = raw
        else:
            raise ValueError(f"Unsupported activation function: {layer_data['activation_function']}")
        
        layer = Layer(
            size=layer_data["size"],
            input_size=layer_data["input_size"],
            activation_function=activation_function
        )
        layer.weights = np.array(layer_data["weights"])  # Konvertiere Listen zurück in numpy-Arrays
        layer.biases = np.array(layer_data["biases"])
        net.layers.append(layer)
    
    return net