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
    
    def backward(self, y_true, y_pred, X, learning_rate=0.01, loss_type="cross_entropy", clip_value=None): # X shape (batch_size, input_size); y_true shape (batch_size, output_size)
        batch_size = y_true.shape[0]
        error = y_pred - y_true # This is d(Loss)/d(a_last) for MSE, and d(Loss)/d(z_last) for Softmax+CrossEntropy

        if loss_type == "cross_entropy":
            loss = cross_entropy_loss(y_true, y_pred)
        elif loss_type == "mse":
            loss = mse_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
            
        last_layer = self.layers[-1]

        # For Softmax + CrossEntropy, error = y_pred - y_true is already dL/dz_last
        # For Linear + MSE, error = y_pred - y_true is dL/da_last. Since da_last/dz_last = 1 (for linear), dL/dz_last = error.
        last_layer.dz = error 
        
        prev_a_for_last_layer = self.layers[-2].a if len(self.layers) > 1 else X
        last_layer.dw = np.dot(prev_a_for_last_layer.T, last_layer.dz) / batch_size
        last_layer.db = np.sum(last_layer.dz, axis=0, keepdims=True) / batch_size

        if clip_value is not None:
            last_layer.dw = np.clip(last_layer.dw, -clip_value, clip_value)
            last_layer.db = np.clip(last_layer.db, -clip_value, clip_value)

        last_layer.weights -= learning_rate * last_layer.dw
        last_layer.biases -= learning_rate * last_layer.db

        for i in reversed(range(len(self.layers) - 1)): # Iterate up to the second to last layer
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            prev_a = self.layers[i - 1].a if i > 0 else X

            layer.da = np.dot(next_layer.dz, next_layer.weights.T)
            
            if layer.activation_derivative:
                layer.dz = layer.da * layer.activation_derivative(layer.a)
            else:
                # If no derivative function is explicitly set (e.g., for 'raw' activation),
                # and it's used in a hidden layer (unlikely for 'raw'), assume derivative is 1.
                # This primarily applies if 'raw' was used as hidden layer activation.
                # For output layer 'raw', its dz is handled above.
                layer.dz = layer.da 

            layer.dw = np.dot(prev_a.T, layer.dz) / batch_size
            layer.db = np.sum(layer.dz, axis=0, keepdims=True) / batch_size

            if clip_value is not None:
                layer.dw = np.clip(layer.dw, -clip_value, clip_value)
                layer.db = np.clip(layer.db, -clip_value, clip_value)

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

def mse_loss(y_true, y_pred):
    """
    Calculates the Mean Squared Error loss.
    y_true: true target values
    y_pred: predicted values
    """
    loss = np.mean((y_pred - y_true)**2)
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

def _to_one_hot(state, dimension):
    vec = np.zeros(dimension)
    vec[state] = 1
    return vec

# Ergänze zum Beispiel eine Q-Learning-Trainingsmethode:
def train_q_learning(env, episodes, alpha=0.01, gamma=0.99, initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay_rate=0.995, clip_grad_value=1.0, save_interval=0, filename=None): # Added save_interval and filename
    epsilon = initial_epsilon
    # Optional: Liste für das Plotten von Belohnungen
    # total_rewards_per_episode = []

    for episode in range(episodes):
        result = env.reset()
        if isinstance(result, tuple):
            state, info = result
        else:
            state = result

        done = False
        current_episode_total_reward = 0

        # Fortschrittsanzeige seltener, um die Konsole nicht zu überfluten
        if episode % 50 == 0:
            print(f"Episode {episode}/{episodes}, Epsilon: {epsilon:.4f}")

        while not done:
            state_vec = _to_one_hot(state, env.observation_space.n)
            q_values = net.forward(state_vec.reshape(1, -1))

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_values)

            next_state, reward, done, info = env.step(action)
            current_episode_total_reward += reward
            
            next_state_vec = _to_one_hot(next_state, env.observation_space.n)
            next_q = net.forward(next_state_vec.reshape(1, -1))

            target_q = q_values.copy()
            if done:
                target_q[0, action] = reward
            else:
                target_q[0, action] = reward + gamma * np.max(next_q)

            # Stelle sicher, dass state_vec die korrekte Form für X in backward hat
            # Call backward with loss_type="mse" for Q-learning and gradient clipping
            net.backward(target_q, q_values, state_vec.reshape(1, -1), alpha, loss_type="mse", clip_value=clip_grad_value)
            state = next_state
        
        # total_rewards_per_episode.append(current_episode_total_reward)

        # Epsilon-Decay
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay_rate
        # Alternativ: epsilon = max(min_epsilon, epsilon - decay_value_per_episode)

        # Autosave logic
        if save_interval > 0 and filename and (episode + 1) % save_interval == 0:
            save_network(filename)
            print(f"Network saved at episode {episode + 1}")

    print(f"Training finished. Final Epsilon: {epsilon:.4f}")
    # return total_rewards_per_episode # Optional zurückgeben für Analyse