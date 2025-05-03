import numpy as np

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
            print(f"Layer {i} output: {layer.a}")
        return a
    
    def backward(self, y_true, y_pred, X, learning_rate=0.01): # X shape (batch_size, input_size); y_true shape (batch_size, output_size)
        batch_size = y_true.shape[0]
        error = y_pred - y_true
        loss = cross_entropy_loss(y_true, y_pred)
        last_layer = self.layers[-1]

        last_layer.dz = error
        last_layer.dw = np.dot(self.layers[-2].a.T, last_layer.dz) / batch_size
        last_layer.db = np.sum(last_layer.dz, axis=0, keepdims=True) / batch_size

        for i in reversed(range(len(self.layers) -1)):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            prev_a = self.layers[i - 1].a if i > 0 else X

            # calculate gradients
            layer.da = np.dot(next_layer.dz, next_layer.weights.T)
            layer.dz = layer.da * layer.activation_derivative(layer.z)
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
    net.addLayer(input_size, input_size, raw) # Input layer
    
    for segment in hidden_layers: # hidden layers
        if segment[1] == "sigmoid":
            activation_function = sigmoid
        elif segment[1] == "relu":
            activation_function = relu
        for i in range(segment[2]):
            net.addLayer(segment[0], net.layers[-1].size, activation_function)
    
    net.addLayer(output_size, net.layers[-1].size, softmax) # Output layer

def get_network():
    return net

def train(X, y, epochs, batch_size, learning_rate):
    pass