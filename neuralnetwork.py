import numpy as np
import json
import os
import random
from collections import deque

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
            raise ValueError(f"Input shape {inputs.shape} does not match layer input size {self.layers[0].input_size}")
        x = inputs

        for layer in self.layers: # Iterate through actual layer objects
            layer.z = np.dot(x, layer.weights) + layer.biases
            layer.a = layer.activation_function(layer.z)
            x = layer.a # Output of current layer is input to next
        return x # Return the activation of the last layer
    
    def backward(self, y_true, y_pred, X, learning_rate=0.01, loss_type="cross_entropy", clip_value=None): # X shape (batch_size, input_size); y_true shape (batch_size, output_size)
        batch_size = y_true.shape[0]
        
        # Calculate initial error (dL/dz_last or dL/da_last)
        if loss_type == "cross_entropy": # Typically used with softmax output
            # For Softmax + CrossEntropy, y_pred - y_true is dL/dz_last
            error = y_pred - y_true 
        elif loss_type == "mse": # Typically used with linear output for regression/Q-values
            # For Linear + MSE, y_pred - y_true is dL/da_last.
            # Since (da_last/dz_last) = 1 for linear, dL/dz_last = dL/da_last * 1 = y_pred - y_true
            error = y_pred - y_true
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
            
        last_layer = self.layers[-1]
        last_layer.dz = error # This is now dL/dz for the last layer

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
                layer.dz = layer.da * layer.activation_derivative(layer.a) # Use stored 'a'
            else:
                # This case should ideally only be for linear layers or if derivative is 1
                # For Q-learning output layer (if linear), derivative is 1, so dz = da.
                # If a hidden layer has no derivative set, it's an issue.
                layer.dz = layer.da 

            layer.dw = np.dot(prev_a.T, layer.dz) / batch_size
            layer.db = np.sum(layer.dz, axis=0, keepdims=True) / batch_size

            if clip_value is not None:
                layer.dw = np.clip(layer.dw, -clip_value, clip_value)
                layer.db = np.clip(layer.db, -clip_value, clip_value)

            layer.weights -= learning_rate * layer.dw
            layer.biases -= learning_rate * layer.db
        
        # Recalculate loss based on the type (optional, if needed outside)
        if loss_type == "cross_entropy":
            loss = cross_entropy_loss(y_true, y_pred) # y_pred here is a_last
        if loss_type == "mse":
            loss = mse_loss(y_true, y_pred) # y_pred here is a_last
        return loss
    
    def clone(self):
        # Create a deep copy of the network (for target network)
        clone_net = Network()
        for layer in self.layers:
            new_layer = Layer(layer.size, layer.input_size, layer.activation_function)
            new_layer.weights = np.copy(layer.weights)
            new_layer.biases = np.copy(layer.biases)
            clone_net.layers.append(new_layer)
        return clone_net

    def set_weights(self, other_net):
        # Copy weights from another network
        for l_self, l_other in zip(self.layers, other_net.layers):
            l_self.weights = np.copy(l_other.weights)
            l_self.biases = np.copy(l_other.biases)

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
    clear_network()
    # hiddenlayers: [size, activation_function_str, amount]
    global net 
    # It's good practice to clear the network if this function can be called multiple times
    # to redefine it, or ensure it's only called on an empty net.
    # clear_network() # Optional: uncomment if you want create_network to always start fresh

    current_processing_input_size = input_size

    for segment in hidden_layers: # hidden layers
        size, activation_function_str, amount = segment
        
        activation_fn = None
        if activation_function_str == "sigmoid":
            activation_fn = sigmoid
        elif activation_function_str == "relu":
            activation_fn = relu
        else:
            raise ValueError(f"Unsupported activation string {activation_function_str}")

        print(segment) # e.g. (64, 'relu', 1)
        for _ in range(amount):
            net.addLayer(size, current_processing_input_size, activation_fn)
            current_processing_input_size = size # Output of this layer is input to next
    
    # Output layer for Q-values should be linear
    net.addLayer(output_size, current_processing_input_size, raw) # Use raw (linear) activation
    print(f"Network created with input size {input_size}, output size {output_size}, and hidden layers {hidden_layers}")

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
    clear_network()
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

def convert_structure(state):
    vec = []
    for i in range(len(state)):
        for cell in state[i]:
            vec.append(cell)
    return np.array(vec)

def q_train(env, episodes, learning_rate=0.001, gamma=0.99, epsilon=1, min_epsilon=0.01, epsilon_decay=0.995, clip_grad_value=0.5, save_interval=0, filename=None, max_steps=1000, target_update_freq=20):
    replay_buffer = deque(maxlen=10000)
    batch_size = 32
    global net
    target_net = net.clone()  # Initialize target network

    for episode in range(episodes):
        state = env.reset()
        done = False
        step_count = 0
        while not done:
            state_vec = convert_structure(state) 
            q_values = net.forward(state_vec.reshape(1, -1))

            if np.random.rand() < epsilon:
                action = env.action_space_sample() # Use the method directly
            else:
                action = np.argmax(q_values)

            next_state, reward, done, info = env.step(action)
            state_vec = convert_structure(state)
            next_state_vec = convert_structure(next_state)
            replay_buffer.append((state_vec, action, reward, next_state_vec, done))
            state = next_state

            # Only train if enough samples in buffer
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                for state_b, action_b, reward_b, next_state_b, done_b in batch:
                    # Compute target Q-value using target network
                    q_values = net.forward(state_b.reshape(1, -1))
                    next_q_values = target_net.forward(next_state_b.reshape(1, -1))
                    target = q_values.copy()
                    if done_b:
                        target[0, action_b] = reward_b
                    else:
                        target[0, action_b] = reward_b + gamma * np.max(next_q_values)
                    # Backpropagate
                    loss = net.backward(target, q_values, state_b.reshape(1, -1), learning_rate, loss_type="mse", clip_value=clip_grad_value)
                    total_reward = reward_b
            step_count += 1
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Update target network every target_update_freq episodes
        if (episode + 1) % target_update_freq == 0:
            target_net.set_weights(net)

        # Print average loss for the episode if available
        if 'loss' in locals():
            print(f"Episode {episode}/{episodes}, Loss: {loss}, Total Reward: {locals().get('total_reward', 'N/A')}, Epsilon: {epsilon:.4f}, Steps: {step_count}")
        else:
            print(f"Episode {episode}/{episodes}, Total Reward: {locals().get('total_reward', 'N/A')}, Epsilon: {epsilon:.4f}, Steps: {step_count}")

        if save_interval > 0 and filename and (episode + 1) % save_interval == 0:
            save_network(filename)
            print(f"Network saved at episode {episode + 1}")


# Ergänze zum Beispiel eine Q-Learning-Trainingsmethode:
# def train_q_learning(env, episodes, alpha=0.01, gamma=0.99, initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay_rate=0.995, clip_grad_value=1.0, save_interval=0, filename=None):
#     epsilon = initial_epsilon
#     # Optional: Liste für das Plotten von Belohnungen
#     # total_rewards_per_episode = []

#     for episode in range(episodes):
#         result = env.reset()
#         if isinstance(result, tuple):
#             state, info = result
#         else:
#             state = result

#         done = False
#         current_episode_total_reward = 0

#         # Fortschrittsanzeige seltener, um die Konsole nicht zu überfluten
#         if episode % 50 == 0:
#             print(f"Episode {episode}/{episodes}, Epsilon: {epsilon:.4f}")

#         while not done:
#             state_vec = _to_one_hot(state, env.observation_space.n)
#             q_values = net.forward(state_vec.reshape(1, -1))

#             if np.random.rand() < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 action = np.argmax(q_values)

#             next_state, reward, done, info = env.step(action)
#             current_episode_total_reward += reward
            
#             next_state_vec = _to_one_hot(next_state, env.observation_space.n)
#             next_q = net.forward(next_state_vec.reshape(1, -1))

#             target_q = q_values.copy()
#             if done:
#                 target_q[0, action] = reward
#             else:
#                 target_q[0, action] = reward + gamma * np.max(next_q)

#             # Stelle sicher, dass state_vec die korrekte Form für X in backward hat
#             # Call backward with loss_type="mse" for Q-learning and gradient clipping
#             net.backward(target_q, q_values, state_vec.reshape(1, -1), alpha, loss_type="mse", clip_value=clip_grad_value)
#             state = next_state
        
#         # total_rewards_per_episode.append(current_episode_total_reward)

#         # Epsilon-Decay
#         if epsilon > min_epsilon:
#             epsilon *= epsilon_decay_rate
#         # Alternativ: epsilon = max(min_epsilon, epsilon - decay_value_per_episode)

#         # Autosave logic
#         if save_interval > 0 and filename and (episode + 1) % save_interval == 0:
#             save_network(filename)
#             print(f"Network saved at episode {episode + 1}")

#     print(f"Training finished. Final Epsilon: {epsilon:.4f}")
#     # return total_rewards_per_episode # Optional zurückgeben für Analyse