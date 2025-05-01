import numpy as np

class Layer:
    def __init__(self, size, input_size, activation_function):
        self.size = size
        self.input_size = input_size
        self.activation_function = activation_function

        self.weights = np.random.randn(input_size, size) * 0.01
        self.biases = np.zeros((1, size))
        self.output = None

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
        x = inputs

        for i in range(len(self.layers)):
            layer = self.layers[i]
            z = np.dot(x, layer.weights) + layer.biases
            x = layer.activation_function(z)
            layer.output = x
            print(f"Layer {i} output: {layer.output}")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)

# --- Network shape creator --- #
create_or_load = input("Create or load network shape (create/load): ")
if create_or_load == "load":
    with open("networkshape.txt", "r") as f:
        lines = f.readlines()
        net = Network()
        for line in lines:
            size, input_size, activation_function = line.strip().split()
            size = int(size)
            input_size = int(input_size)
            if activation_function == "sigmoid":
                activation_function = sigmoid
            elif activation_function == "relu":
                activation_function = relu
            else:
                raise ValueError("Invalid activation function")
            net.addLayer(size, input_size, activation_function)
    print("Network shape loaded")
elif create_or_load == "create":
    net = Network()

    while True:
        create_layer = input("Create layer? (y/n): ")
        if create_layer == "y":
            size = int(input("Layer size: "))
            if len(net.layers) == 0:
                input_size = int(input("Input size: "))
            else:
                input_size = net.layers[-1].size
            activation_function = input("Activation function (sigmoid/relu): ")
            if activation_function == "sigmoid":
                activation_function = sigmoid
            elif activation_function == "relu":
                activation_function = relu
            else:
                print("Invalid activation function")
                continue

            net.addLayer(size, input_size, activation_function)
        elif create_layer == "n":
            break
        else:
            print("Invalid input")
    print("Network shape created")

print("Layers:")
for i, layer in enumerate(net.layers):
    print(f"Layer {i}: size={layer.size}, input_size={layer.input_size}, activation_function={layer.activation_function.__name__}")
save = input("Save network shape? (y/n): ")
if save == "y":
    with open("networkshape.txt", "w") as f:
        for layer in net.layers:
            f.write(f"{layer.size} {layer.input_size} {layer.activation_function.__name__}\n")
    print("Network shape saved to networkshape.txt")