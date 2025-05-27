import neuralnetwork as nn
import numpy as np
import math
import os
import time
import random
import matplotlib.pyplot as plt
import PIL.Image
import scipy.ndimage as ndi # Added for data augmentation

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


def augment_image(image_vector, image_shape=(28, 28)):
    image = image_vector.copy().reshape(image_shape) # Use copy to avoid modifying original X_train

    # 1. Random Rotation
    if random.random() < 0.75: # Apply 75% of the time
        angle = random.uniform(-10, 10) # Rotate between -10 and 10 degrees
        image = ndi.rotate(image, angle, reshape=False, mode='constant', cval=0.0, order=1) # cval=0.0 for black background on normalized image

    # 2. Random Shift
    if random.random() < 0.75: # Apply 75% of the time
        shift_h = random.uniform(-2, 2) # Shift vertically by -2 to 2 pixels
        shift_w = random.uniform(-2, 2) # Shift horizontally by -2 to 2 pixels
        image = ndi.shift(image, (shift_h, shift_w), mode='constant', cval=0.0, order=1)

    # 3. Random Zoom
    if random.random() < 0.5: # Apply 50% of the time
        zoom_factor = random.uniform(0.9, 1.1) # Zoom between 90% and 110%
        h, w = image.shape
        
        zoomed_image = ndi.zoom(image, zoom_factor, mode='constant', cval=0.0, order=1)
        zh, zw = zoomed_image.shape
        
        processed_image = np.zeros_like(image) # Create an empty canvas of original size
        
        if zoom_factor < 1.0: # Image shrunk, place it in the center
            y_start = (h - zh) // 2
            x_start = (w - zw) // 2
            y_end = y_start + zh
            x_end = x_start + zw
            if y_start >= 0 and y_end <= h and x_start >= 0 and x_end <= w: # Check bounds
                 processed_image[y_start:y_end, x_start:x_end] = zoomed_image
            else: # Fallback if bounds are off (e.g. extreme zoom out)
                processed_image = image # No change
        else: # Image grew, crop the center
            y_start_crop = (zh - h) // 2
            x_start_crop = (zw - w) // 2
            y_end_crop = y_start_crop + h
            x_end_crop = x_start_crop + w
            if y_start_crop >=0 and y_end_crop <= zh and x_start_crop >=0 and x_end_crop <= zw: # Check bounds
                processed_image = zoomed_image[y_start_crop:y_end_crop, x_start_crop:x_end_crop]
            else: # Fallback
                processed_image = image # No change
        
        image = processed_image
        # Ensure final shape is correct, resize if necessary (e.g. due to rounding in // operations)
        if image.shape != image_shape:
            img_pil = PIL.Image.fromarray((image * 255).astype(np.uint8)) # Convert to 0-255 for PIL
            img_pil = img_pil.resize(image_shape[::-1], PIL.Image.Resampling.LANCZOS) # PIL resize takes (width, height)
            image = np.array(img_pil) / 255.0

    image = np.clip(image, 0.0, 1.0) # Ensure values are still in [0, 1]
    return image.reshape(-1) # Flatten back to vector


while True:
    command = input("Command: ")

    if command == "exit":
        break
    elif command == "create":
        hidden_layers = [
            [128, "relu", 1]
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
        use_augmentation = input("Use data augmentation? (y/n): ").lower() == 'y'
        save_interval = int(input("Save interval in epochs (0 to disable saving): "))
        if save_interval > 0:
            filename = input("Filename for saving model: ")
            filename = "NeuralNetworks/" + filename + ".json"
        else:
            filename = None
        epochs = int(input("Epochs: "))
        batch_size = int(input("Batch size: "))
        learning_rate = float(input("Learning rate (e.g., 0.001): ")) # Get learning rate from user
        
        current_net = nn.get_network()
        if current_net is None:
            print("Error: Network not created. Please use the 'create' command first.")
            continue

        start_time = time.time()
        print("Training started...")
        
        for epoch in range(epochs):
            epoch_loss = 0
            # Shuffle data at the beginning of each epoch
            permutation = np.random.permutation(X_train.shape[0])
            X_epoch_shuffled = X_train[permutation]
            y_epoch_shuffled = y_train_one_hot[permutation]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch_original = X_epoch_shuffled[i:i + batch_size]
                y_batch = y_epoch_shuffled[i:i + batch_size]

                if use_augmentation:
                    # Augment the batch
                    X_batch_processed = np.array([augment_image(img_vec) for img_vec in X_batch_original])
                else:
                    X_batch_processed = X_batch_original
                
                # Forward pass
                y_pred = current_net.forward(X_batch_processed)

                # Backward pass
                loss = current_net.backward(y_batch, y_pred, X_batch_processed, learning_rate)
                epoch_loss += loss * X_batch_processed.shape[0] # Accumulate weighted loss for the epoch

            avg_epoch_loss = epoch_loss / X_train.shape[0]
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
            if save_interval > 0 and filename and (epoch + 1) % save_interval == 0:
                nn.save_network(filename)
                print(f"Network saved at epoch {epoch + 1}")
        
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60
        print(f"Training completed in {elapsed_minutes:.2f} minutes.")

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
    elif command == "fulltest":
        amount = int(input("Amount of samples to test: "))
        accuracies = []
        for i in range(amount):
            index = random.randint(0, X_test.shape[0] -1)
            X = X_test[index].reshape(1, -1)
            y = y_test_one_hot[index].reshape(1, -1)
            accuracy, predictions = nn.test(X, y)
            predictions = np.round(predictions, 2)
            accuracies.append(accuracy)

            print(f"Predicted label: {np.argmax(predictions)}")
            print(f"Correct label: {np.argmax(y)}")
            print(f"Test accuracy: {accuracy*100}%\n")

        average_accuracy = sum(accuracies) / len(accuracies)
        print(f"Average accuracy over {amount} samples: {average_accuracy * 100:.2f}%")
        
    elif command == "customtest":
        test_all = input("Test all images in folder? (y/n): ")
        if test_all.lower() == "y":
            folder_path = "./customTests/"
            accuracies = []

            for filename in os.listdir(folder_path):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    img_path = os.path.join(folder_path, filename)
                    img = PIL.Image.open(img_path).convert("L").resize((28, 28))
                    img_array = np.array(img) / 255.0
                    X = img_array.reshape(1, -1)
                    # Dummy label (not used for prediction)
                    y = np.zeros((1, 10))
                    accuracy, predictions = nn.test(X, y)
                    predictions = np.round(predictions, 2)
                    if np.argmax(predictions) == int(filename[0]):
                        accuracies.append(1)
                    else:
                        accuracies.append(0)
                    print(f"Predictions for {filename}: {predictions}")
                    print(f"Predicted label: {np.argmax(predictions)}\n")
            average_accuracy = sum(accuracies) / len(accuracies)
            print(f"Average accuracy: {average_accuracy * 100:.2f}%")
        else:
            path = input("Path to image: ")
            path = "./customTests/" + path
            img = PIL.Image.open(path).convert("L").resize((28, 28))
            img_array = np.array(img) / 255.0
            X = img_array.reshape(1, -1)
            # Dummy label (not used for prediction)
            y = np.zeros((1, 10))
            accuracy, predictions = nn.test(X, y)
            predictions = np.round(predictions, 2)
            print(f"Predictions: {predictions}")
            print(f"Predicted label: {np.argmax(predictions)}")
            plt.imshow(img_array, cmap='gray')
            plt.title("Custom Image")
            plt.axis('off')
            plt.show()
    else:
        print("Command unknown.")