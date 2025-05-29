import gym
import neuralnetwork as nn
import numpy as np # Korrigierter Import
import time
import os

# Global environment for Taxi-v3, primarily for training (no rendering needed here)
# Die alte API wird standardmäßig verwendet, passend zu nn.train_q_learning
env_train = gym.make("Taxi-v3") 
# Um die neue API zu verwenden (optional, erfordert Code-Anpassungen in step-Aufrufen):
# env_train = gym.make("Taxi-v3", new_step_api=True)

state_dim = env_train.observation_space.n
action_dim = env_train.action_space.n

def test_agent_visual(environment, network, num_episodes=5):
    """
    Führt einen visualisierten Test des trainierten Agenten durch.
    :param environment: Die Gym-Umgebung.
    :param network: Das trainierte neuronale Netzwerk (das 'net'-Objekt).
    :param num_episodes: Anzahl der Testepisoden.
    """
    if network is None or not network.layers:
        print("Error: No network available for testing. Please create or load a network.")
        return

    for episode in range(num_episodes):
        result = environment.reset()
        if isinstance(result, tuple):
            current_state, info = result
        else:
            current_state = result

        print(f"\nStarting visualization episode {episode + 1}")
        try:
            environment.render()
        except Exception as e:
            print(f"Warning: Could not render environment (might be a headless server or missing display): {e}")
            # pass # Besser, hier keine Exception zu unterdrücken, wenn render() kritisch ist
        time.sleep(0.2) # Pause am Anfang der Episode kann bleiben

        done = False
        total_reward = 0
        steps = 0

        while not done:
            state_vector = nn._to_one_hot(current_state, environment.observation_space.n)
            q_values_for_state = network.forward(state_vector.reshape(1, -1))
            print(f"State: {current_state}, Q-Values: {q_values_for_state}")
            action_to_take = np.argmax(q_values_for_state)

            next_state, reward, done, info = environment.step(action_to_take)

            total_reward += reward
            current_state = next_state
            steps += 1
            
            try:
                environment.render()
            except Exception:
                # Hier könnte man auch loggen, statt still zu scheitern
                pass 
            print(f"Step: {steps}, Action: {action_to_take}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
            # time.sleep(0.05) # Testweise entfernen oder stark reduzieren, z.B. auf 0.01
            # Für maximale Reaktionsfähigkeit des Fensters, entferne es ganz.
            # Wenn die Simulation zu schnell ist, versuche einen sehr kleinen Wert:
            # time.sleep(0.001) # Sehr kurze Pause

            if done:
                print(f"Episode {episode + 1} finished after {steps} steps. Total reward: {total_reward:.2f}")
                time.sleep(0.5) # Pause am Ende der Episode kann bleiben
                break
    try:
        environment.close()
    except Exception:
        pass

def main():
    # Sicherstellen, dass das Verzeichnis für gespeicherte Netzwerke existiert
    if not os.path.exists("NeuralNetworks"):
        os.makedirs("NeuralNetworks")

    while True:
        command = input("Command: ").strip().lower()

        if command == "exit":
            print("Exiting Taxi-v3 controller.")
            break
        elif command == "create":
            nn.clear_network() 
            # Standardkonfiguration für Taxi-v3 Q-Learning Netzwerk
            # Input: state_dim (500), Output: action_dim (6)
            # Für Q-Learning ist die letzte Schicht typischerweise linear (raw).
            # Die create_network Funktion in neuralnetwork.py fügt standardmäßig Softmax hinzu.
            # Wir erstellen das Netzwerk und passen dann die letzte Schicht an.
            hidden_layers_config = [(64, "relu", 1)]
            print(f"Creating network for Taxi-v3: input_size={state_dim}, output_size={action_dim}")
            nn.create_network(input_size=state_dim, output_size=action_dim, hidden_layers=hidden_layers_config)
            
            # Anpassung der letzten Schicht für Q-Learning (lineare Aktivierung)
            current_net = nn.get_network()
            if current_net and current_net.layers:
                print(f"Original output layer activation: {current_net.layers[-1].activation_function.__name__}")
                current_net.layers[-1].activation_function = nn.raw # Setze auf lineare Aktivierung
                current_net.layers[-1].activation_derivative = None # Keine Ableitung für 'raw'
                print(f"Output layer activation changed to: {current_net.layers[-1].activation_function.__name__}")
            print("Network created.")

        elif command == "clear":
            nn.clear_network()
            print("Network cleared.")
        elif command == "show":
            network = nn.get_network()
            if network is not None and network.layers:
                print("Current network structure:")
                for i, layer in enumerate(network.layers):
                    print(f"Layer {i}: Neurons={layer.size}, Activation='{layer.activation_function.__name__}', Inputs={layer.input_size}")
            else:
                print("No network is currently created or loaded.")
        elif command == "train":
            save_interval = int(input("Saveinterval (0 to disable saving): "))
            if save_interval > 0:
                filename_input = input("Enter filename for saving (e.g., taxi_q_model): ").strip()
                if not filename_input:
                    print("Filename cannot be empty.")
                    continue
                filepath = os.path.join("NeuralNetworks", filename_input + ".json")
            network = nn.get_network()
            if not (network and network.layers):
                print("No network to train. Please use 'create' or 'load' first.")
                continue
            try:
                episodes = int(input("Enter number of episodes for training (e.g., 10000): "))
                alpha = float(input("Enter learning rate (alpha, e.g., 0.01): "))
                gamma = float(input("Enter discount factor (gamma, e.g., 0.99): "))
                initial_epsilon = float(input("Enter initial exploration rate (epsilon, e.g., 1.0): "))
                min_epsilon = float(input("Enter minimum exploration rate (e.g., 0.01): "))
                epsilon_decay_rate = float(input("Enter epsilon decay rate (e.g., 0.995, must be < 1): "))
                
                if not (0 < epsilon_decay_rate < 1):
                    print("Epsilon decay rate must be between 0 and 1 (exclusive). Using default 0.995.")
                    epsilon_decay_rate = 0.995

                print("Starting Q-learning training...")
                start_time = time.time()
                # Verwende env_train für das Training
                nn.train_q_learning(env_train, 
                                    episodes=episodes, 
                                    alpha=alpha, 
                                    gamma=gamma, 
                                    initial_epsilon=initial_epsilon, 
                                    min_epsilon=min_epsilon, 
                                    epsilon_decay_rate=epsilon_decay_rate,
                                    save_interval=save_interval,
                                    filename=filepath)
                end_time = time.time()
                duration_seconds = end_time - start_time
                minutes = int(duration_seconds // 60)
                seconds = int(duration_seconds % 60)
                print(f"Training completed in {minutes} minutes and {seconds} seconds.")
            except ValueError:
                print("Invalid input. Please ensure you enter numbers for training parameters.")
            except Exception as e:
                print(f"An error occurred during training: {e}")

        elif command == "test":
            network = nn.get_network()
            if not (network and network.layers):
                print("No network to test. Please use 'create' or 'load' first.")
                continue
            try:
                num_episodes_test = int(input("Enter number of episodes for testing (e.g., 3): "))
                # Erstelle eine separate Umgebung für den visualisierten Test
                env_visual_test = gym.make("Taxi-v3", render_mode="human")
                test_agent_visual(env_visual_test, network, num_episodes=num_episodes_test)
                env_visual_test.close() # Schließe die Test-Umgebung
            except ValueError:
                print("Invalid input. Please enter a whole number for episodes.")
            except Exception as e:
                print(f"An error occurred during testing: {e}")

        elif command == "save":
            network = nn.get_network()
            if not (network and network.layers):
                print("No network to save. Please use 'create' or 'load' first.")
                continue
            filename_input = input("Enter filename for saving (e.g., taxi_q_model): ").strip()
            if not filename_input:
                print("Filename cannot be empty.")
                continue
            filepath = os.path.join("NeuralNetworks", filename_input + ".json")
            try:
                nn.save_network(filepath)
                print(f"Network successfully saved to {filepath}")
            except Exception as e:
                print(f"Failed to save network: {e}")

        elif command == "load":
            filename_input = input("Enter filename to load (e.g., taxi_q_model): ").strip()
            if not filename_input:
                print("Filename cannot be empty.")
                continue
            filepath = os.path.join("NeuralNetworks", filename_input + ".json")
            if not os.path.exists(filepath):
                print(f"Error: File not found at {filepath}")
                continue
            try:
                nn.clear_network() # Clear current network before loading a new one
                loaded_net = nn.load_network(filepath)
                # Die nn.load_network Funktion modifiziert das globale nn.net Objekt.
                # Ein explizites Zuweisen ist nicht nötig, wenn nn.net global in neuralnetwork.py ist.
                if nn.get_network() and nn.get_network().layers: # Überprüfen, ob das Laden erfolgreich war
                    print(f"Network successfully loaded from {filepath}")
                else:
                    print(f"Failed to load network from {filepath} or the loaded network is empty.")
            except Exception as e:
                print(f"Failed to load network: {e}")
        else:
            print("Unknown command. Available commands: create, clear, show, train, test, save, load, exit")

if __name__ == "__main__":
    main()