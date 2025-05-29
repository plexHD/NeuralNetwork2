import neuralnetwork
import maze
import numpy as np
import time

game = maze.Maze((8, 8))
while True:
    command = input("Command: ").lower().strip()
    if command == "exit":
        print("Exiting...")
        break
    elif command == "render":
        game.render()
    elif command == "reset":
        game = maze.Maze((8, 8))
        game.render()
    elif command == "play":
        while True:
            direction = input("Enter direction (w, e, n, s): ").strip().lower()
            if direction == "exit":
                print("Ending play...")
                break
            if direction in ["w", "e", "n", "s"]:
                game.movePlayer(direction)
                game.render()
            else:
                print("Invalid direction! Use 'w', 'e', 'n', or 's'.")

    elif command == "train":
        # Parameters for Q-learning
        network_filename = "./NeuralNetworks/maze100.json"

        load = input("Load existing network? (y/n): ").strip().lower()
        if load == "n":
            neuralnetwork.create_network(game.observation_space_n + 2, 
                                         game.action_space_n, 
                                         hidden_layers=[(128, "relu", 1)]) # Example: 1 hidden layer of 64 neurons
        else:
            neuralnetwork.load_network(network_filename)


        episodes = int(input("Episodes: "))
        learning_rate = 0.7 # Alpha
        gamma = 0.9         # Discount factor
        epsilon = 1.0       # Exploration rate
        min_epsilon = 0.05
        epsilon_decay = 0.9995
        save_interval = 500
        max_steps = 150

        # Ensure the network is created if not loaded
        # The input size is the number of possible states (observation_space_n)
        # The output size is the number of possible actions (action_space_n)
        start_time = time.time()
        print(f"Starting Q-learning training for {episodes} episodes...")
        neuralnetwork.q_train(env=game,
                              episodes=episodes, 
                              learning_rate=learning_rate, 
                              gamma=gamma, 
                              epsilon=epsilon, 
                              min_epsilon=min_epsilon, 
                              epsilon_decay=epsilon_decay, 
                              clip_grad_value=1.0, # Optional: gradient clipping
                              save_interval=save_interval, 
                              filename=network_filename,
                              max_steps=max_steps)
        
        end_time = time.time()
        training_duration = end_time - start_time
        hours = int(training_duration // 3600)
        minutes = int((training_duration % 3600) // 60)
        print(f"Training finished in {hours} hours and {minutes} minutes.")
    
    elif command == "test":
        # Load the trained network
        network_path = "./NeuralNetworks/maze100.json"
        try:
            neuralnetwork.load_network(network_path)
            print(f"Network loaded from {network_path}")
        except FileNotFoundError:
            print(f"Error: Network file not found at {network_path}. Train the network first.")
            # Instead of returning, we'll just skip the test execution for this command
            # and let the main loop continue to the next command input.
            continue 

        print("Testing the trained Q-learning agent...")
        state = game.reset()
        game.render()
        done = False
        total_reward = 0
        max_steps = 50
        for step in range(max_steps):
            # Convert state to one-hot encoded vector
            state_one_hot = neuralnetwork.convert_structure(state)
            # Reshape for the network (batch size of 1)
            state_input = state_one_hot.reshape(1, -1).astype(np.float32)

            q_values = neuralnetwork.net.forward(state_input)
            action = np.argmax(q_values[0])  # Choose action with highest Q-value
            
            next_state, reward, done, _ = game.step(action)
            game.render()
            print(f"Step {step+1}: Action {action}, Q-values {q_values[0]}, Reward {reward}")
            total_reward += reward
            state = next_state
            
            if done:
                print(f"Episode finished after {step+1} steps.")
                print(f"Total reward: {total_reward}")
                break
            
            # Add a small delay to make the visualization easier to follow
            time.sleep(0.1)
        
        if not done:
            print(f"Testing finished after {max_steps} steps (max_steps reached).")
            print(f"Total reward: {total_reward}")
        game.reset() # Reset for next command
    else:
        print("Unknown command.")