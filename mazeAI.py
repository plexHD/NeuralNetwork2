import neuralnetwork
import maze
import numpy as np
import time
import hyperopt
import json
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
maze_size = (8, 8)
game = maze.Maze(maze_size)
while True:
    command = input("Command: ").lower().strip()
    if command == "exit":
        print("Exiting...")
        break
    elif command == "render":
        game.render()
    elif command == "reset":
        game = maze.Maze(maze_size)
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
                                         hidden_layers=[(64, "relu", 1)]) # Example: 1 hidden layer of 64 neurons
        else:
            neuralnetwork.load_network(network_filename)


        episodes = int(input("Episodes: "))
        learning_rate = 0.0005
        gamma = 0.84
        epsilon = 0.77
        min_epsilon = 0.13
        epsilon_decay = 0.9992
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
    elif command == "hyperopt":
        def hyperopt_objective(params):
            # Unpack hyperparameters
            learning_rate = params['learning_rate']
            gamma = params['gamma']
            epsilon = params['epsilon']
            min_epsilon = params['min_epsilon']
            epsilon_decay = params['epsilon_decay']
            hidden_size = int(params['hidden_size'])

            # Re-create the environment and network for each trial
            game = maze.Maze((8, 8))
            neuralnetwork.create_network(game.observation_space_n + 2, game.action_space_n, hidden_layers=[(hidden_size, "relu", 1)])

            # Run a short training (e.g., 200 episodes)
            neuralnetwork.q_train(
                env=game,
                episodes=200,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                min_epsilon=min_epsilon,
                epsilon_decay=epsilon_decay,
                clip_grad_value=1.0,
                save_interval=0,
                filename=None,
                max_steps=100
            )

            # Evaluate performance (average reward over 5 test episodes)
            total_rewards = []
            for _ in range(5):
                state = game.reset()
                done = False
                total_reward = 0
                steps = 0
                while not done and steps < 50:
                    state_one_hot = neuralnetwork.convert_structure(state)
                    state_input = state_one_hot.reshape(1, -1).astype(np.float32)
                    q_values = neuralnetwork.net.forward(state_input)
                    action = np.argmax(q_values[0])
                    next_state, reward, done, _ = game.step(action)
                    total_reward += reward
                    state = next_state
                    steps += 1
                total_rewards.append(total_reward)
            avg_reward = np.mean(total_rewards)
            # We want to maximize reward, but hyperopt minimizes, so return -avg_reward
            return {'loss': -avg_reward, 'status': STATUS_OK}

        print("Starting hyperparameter optimization with hyperopt...")
        space = {
            'learning_rate': hp.uniform('learning_rate', 0.01, 1.0),
            'gamma': hp.uniform('gamma', 0.7, 0.99),
            'epsilon': hp.uniform('epsilon', 0.7, 1.0),
            'min_epsilon': hp.uniform('min_epsilon', 0.01, 0.2),
            'epsilon_decay': hp.uniform('epsilon_decay', 0.995, 0.9999),
            'hidden_size': hp.quniform('hidden_size', 32, 256, 16)
        }
        trials = Trials()
        best = fmin(
            fn=hyperopt_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials
        )
        print("Best hyperparameters found:")
        print(best)
    elif command.startswith("savemaze"):
        parts = command.split()
        if len(parts) < 2:
            print("Usage: savemaze <filename>")
        else:
            filename = parts[1] + ".json"
            maze_data = {
                'size': game.size,
                'structure': game.structure,
                'player_position': game.player_position,
                'start_pos_initial': game.start_pos_initial,
                'end_pos': getattr(game, 'end_pos', None)
            }
            with open(filename, 'w') as f:
                json.dump(maze_data, f)
            print(f"Maze saved to {filename}")
    elif command.startswith("loadmaze"):
        parts = command.split()
        if len(parts) < 2:
            print("Usage: loadmaze <filename>")
        else:
            filename = parts[1] + ".json"
            try:
                with open(filename, 'r') as f:
                    maze_data = json.load(f)
                game = maze.Maze(tuple(maze_data['size']))
                game.structure = maze_data['structure']
                game.player_position = tuple(maze_data['player_position'])
                game.start_pos_initial = tuple(maze_data['start_pos_initial'])
                if maze_data.get('end_pos') is not None:
                    game.end_pos = tuple(maze_data['end_pos'])
                print(f"Maze loaded from {filename}")
                game.render()
            except Exception as e:
                print(f"Failed to load maze: {e}")
    elif command == "pbt":
        # Population Based Training for maze Q-learning
        population_size = 5
        generations = 5
        episodes_per_agent = 300
        max_steps = 150
        population = []
        results = []
        # Define hyperparameter ranges
        lr_range = (0.0005, 0.01)
        gamma_range = (0.7, 0.99)
        epsilon_range = (0.5, 1.0)
        min_epsilon_range = (0.01, 0.2)
        epsilon_decay_range = (0.995, 0.9999)
        hidden_size_range = (32, 128)

        # Initialize population
        for i in range(population_size):
            hp = {
                'learning_rate': np.random.uniform(*lr_range),
                'gamma': np.random.uniform(*gamma_range),
                'epsilon': np.random.uniform(*epsilon_range),
                'min_epsilon': np.random.uniform(*min_epsilon_range),
                'epsilon_decay': np.random.uniform(*epsilon_decay_range),
                'hidden_size': int(np.random.uniform(*hidden_size_range))
            }
            population.append(hp)

        for gen in range(generations):
            print(f"\n--- PBT Generation {gen+1}/{generations} ---")
            gen_results = []
            for idx, hp in enumerate(population):
                print(f"\nTraining agent {idx+1}/{population_size} with hyperparameters: {hp}")
                # Create new environment and network for each agent
                game = maze.Maze((8, 8))
                neuralnetwork.create_network(game.observation_space_n + 2, game.action_space_n, hidden_layers=[(hp['hidden_size'], "relu", 1)])
                # Train agent
                neuralnetwork.q_train(
                    env=game,
                    episodes=episodes_per_agent,
                    learning_rate=hp['learning_rate'],
                    gamma=hp['gamma'],
                    epsilon=hp['epsilon'],
                    min_epsilon=hp['min_epsilon'],
                    epsilon_decay=hp['epsilon_decay'],
                    clip_grad_value=1.0,
                    save_interval=0,
                    filename=None,
                    max_steps=max_steps
                )
                # Evaluate agent
                total_rewards = []
                for _ in range(5):
                    state = game.reset()
                    done = False
                    total_reward = 0
                    steps = 0
                    while not done and steps < 50:
                        state_one_hot = neuralnetwork.convert_structure(state)
                        state_input = state_one_hot.reshape(1, -1).astype(np.float32)
                        q_values = neuralnetwork.net.forward(state_input)
                        action = np.argmax(q_values[0])
                        next_state, reward, done, _ = game.step(action)
                        total_reward += reward
                        state = next_state
                        steps += 1
                    total_rewards.append(total_reward)
                avg_reward = np.mean(total_rewards)
                gen_results.append((avg_reward, hp.copy()))
                print(f"Agent {idx+1} average test reward: {avg_reward}")
            # Sort by performance
            gen_results.sort(key=lambda x: x[0], reverse=True)
            # Exploit: top 2 agents survive, bottom 3 are replaced by mutated copies of top 2
            survivors = [gen_results[0][1], gen_results[1][1]]
            new_population = survivors.copy()
            for i in range(population_size - len(survivors)):
                parent = survivors[i % len(survivors)].copy()
                # Mutate hyperparameters
                parent['learning_rate'] = np.clip(parent['learning_rate'] * np.random.uniform(0.8, 1.2), *lr_range)
                parent['gamma'] = np.clip(parent['gamma'] * np.random.uniform(0.95, 1.05), *gamma_range)
                parent['epsilon'] = np.clip(parent['epsilon'] * np.random.uniform(0.95, 1.05), *epsilon_range)
                parent['min_epsilon'] = np.clip(parent['min_epsilon'] * np.random.uniform(0.95, 1.05), *min_epsilon_range)
                parent['epsilon_decay'] = np.clip(parent['epsilon_decay'] * np.random.uniform(0.98, 1.02), *epsilon_decay_range)
                parent['hidden_size'] = int(np.clip(parent['hidden_size'] + np.random.randint(-8, 8), *hidden_size_range))
                new_population.append(parent)
            population = new_population
            print(f"\nTop agent hyperparameters: {gen_results[0][1]}")
            print(f"Top agent average reward: {gen_results[0][0]}")
        print("\nPBT complete. Best agent hyperparameters:")
        print(gen_results[0][1])
        print(f"Best agent average reward: {gen_results[0][0]}")
    else:
        print("Unknown command.")