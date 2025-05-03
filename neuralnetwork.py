import numpy as np

# --- Neural Network Class --- #
class Layer:
    def __init__(self, size, input_size, activation_function, activation_derivative=None):
        self.size = size
        self.input_size = input_size
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

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

    def addLayer(self, size, input_size, activation_function, activation_derivative):
        layer = Layer(size, input_size, activation_function, activation_derivative)
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


def create_network(shape):
    pass

def train(X, y, epochs, batch_size, learning_rate):
    pass

# # --- Network shape creator --- #
# create_or_load = input("Create or load network shape (create/load): ")
# if create_or_load == "load":
#     with open("networkshape.txt", "r") as f:
#         lines = f.readlines()
#         net = Network()
#         for line in lines:
#             size, input_size, activation_function = line.strip().split()
#             size = int(size)
#             input_size = int(input_size)
#             if activation_function == "sigmoid":
#                 activation_function = sigmoid
#             elif activation_function == "relu":
#                 activation_function = relu
#                 activation_derivative = relu_derivative
#             else:
#                 raise ValueError("Invalid activation function")
#             net.addLayer(size, input_size, activation_function, activation_derivative)
#     print("Network shape loaded")
# elif create_or_load == "create":
#     net = Network()

#     while True:
#         create_layer = input("Create layer? (y/n): ")
#         if create_layer == "y":
#             size = int(input("Layer size: "))
#             if len(net.layers) == 0:
#                 input_size = int(input("Input size: "))
#             else:
#                 input_size = net.layers[-1].size
#             activation_function = input("Activation function (sigmoid/relu): ")
#             if activation_function == "sigmoid":
#                 activation_function = sigmoid
#             elif activation_function == "relu":
#                 activation_function = relu
#                 activation_derivative = relu_derivative
#             else:
#                 print("Invalid activation function")
#                 continue

#             net.addLayer(size, input_size, activation_function, activation_derivative)
#         elif create_layer == "n":
#             break
#         else:
#             print("Invalid input")
#     print("Network shape created")

# print("Layers:")
# for i, layer in enumerate(net.layers):
#     print(f"Layer {i}: size={layer.size}, input_size={layer.input_size}, activation_function={layer.activation_function.__name__}")
# save = input("Save network shape? (y/n): ")
# if save == "y":
#     with open("networkshape.txt", "w") as f:
#         for layer in net.layers:
#             f.write(f"{layer.size} {layer.input_size} {layer.activation_function.__name__}\n")
#     print("Network shape saved to networkshape.txt")