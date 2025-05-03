import neuralnetwork as nn

while True:
    command = input("Command: ")

    if command == "exit":
        break
    elif command == "create":
        hidden_layers = [
            [256, "relu", 2],
            [128, "relu", 1]
        ]
        nn.create_network(784, 10, hidden_layers)

    elif command == "show":
        net = nn.get_network()
        if net is not None:
            print("Network structure:")
            for i, layer in enumerate(net.layers):
                print(f"Layer: {i}, Neurons: {layer.size}, Activation: {layer.activation_function.__name__}")
        else:
            print("Current network is None.")
    else:
        print("Command unkown.")