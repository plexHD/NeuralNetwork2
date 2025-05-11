import neuralnetwork as nn
import numpy as np
import math
from mnist import MNIST
import os
import time
import random
import matplotlib.pyplot as plt

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read the header
        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read the header
        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')
        
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Load MNIST data using custom loader
train_images_path = "C:/Users/mramg/Personal/Code/NeuralNetwork2/archive/train-images.idx3-ubyte"
train_labels_path = "C:/Users/mramg/Personal/Code/NeuralNetwork2/archive/train-labels.idx1-ubyte"
test_images_path = "C:/Users/mramg/Personal/Code/NeuralNetwork2/archive/t10k-images.idx3-ubyte"
test_labels_path = "C:/Users/mramg/Personal/Code/NeuralNetwork2/archive/t10k-labels.idx1-ubyte"

X_train = load_mnist_images(train_images_path)
y_train = load_mnist_labels(train_labels_path)
X_test = load_mnist_images(test_images_path)
y_test = load_mnist_labels(test_labels_path)

# Convert to numpy arrays and normalize data
X_train = np.array(X_train) / 255.0  
y_train = np.array(y_train)
X_test = np.array(X_test) / 255.0
y_test = np.array(y_test)

num_samples = len(X_train)
num_classes = 10
y_train_one_hot = np.zeros((num_samples, num_classes))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

# Correct number of test samples
num_test_samples = len(X_test)

# Create one-hot encoding for test labels
y_test_one_hot = np.zeros((num_test_samples, num_classes))
y_test_one_hot[np.arange(num_test_samples), y_test] = 1

while True:
    command = input("Command: ")

    if command == "exit":
        break
    elif command == "create":
        hidden_layers = [
            [128, "relu", 2]
        ]
        nn.create_network(784, 10, hidden_layers)
    elif command == "clear":
        nn.clear_network()
        print("Network cleared.")
    elif command == "show":
        net = nn.get_network()
        if net is not None:
            print("Network structure:")
            for i, layer in enumerate(net.layers):
                print(f"Layer: {i}, Neurons: {layer.size}, Activation: {layer.activation_function.__name__}, inputs: {layer.input_size}")
        else:
            print("Current network is None.")
    elif command == "save":
        filename = input("Filename: ")
        filename = "NeuralNetworks/" + filename + ".json"

        nn.save_network(filename)
        print(f"Network saved to {filename}.")
    elif command == "load":
        filename = input("Filename: ")
        filename = "NeuralNetworks/" + filename + ".json"
        
        net = nn.load_network(filename)
        print(f"Network loaded from {filename}:")
        for i, layer in enumerate(net.layers):
                print(f"Layer: {i}, Neurons: {layer.size}, Activation: {layer.activation_function.__name__}")
    elif command == "train":
        save_interval = int(input("Save interval in epochs: (0 to disable saving): "))
        if save_interval > 0:
            filename = input("Filename: ")
            filename = "NeuralNetworks/" + filename + ".json"
        else:
            filename = None
        epochs = int(input("Epochs: "))
        batch_size = int(input("Batch size: "))
        start_time = time.time()
        X = X_train
        y = y_train_one_hot
        # print(f"X shape: {X.shape}, y shape: {y.shape}")
        print("Training started...")
        nn.train(X, y, epochs=epochs, batch_size=batch_size, learning_rate=0.001, save_interval=save_interval, filename=filename)
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")

    elif command == "test":
        index = random.randint(0, X_test.shape[0] -1)
        X = X_test[index].reshape(1, -1)
        y = y_test_one_hot[index].reshape(1, -1)
        accuracy, predictions = nn.test(X, y)
        predictions = np.round(predictions, 2)

        print(f"Predictions: {predictions}")
        print(f"Correct labels: {y}\n")

        print(f"Predicted label: {np.argmax(predictions)}")
        print(f"Test accuracy: {accuracy*100}%")

        # Show the image
        plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
        plt.title("Test Image")
        plt.axis('off')
        plt.show()
    else:
        print("Command unknown.")