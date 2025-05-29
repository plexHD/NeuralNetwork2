import pygame
import random
import numpy as np
import neuralnetwork
import time
import flappybird # Import your flappybird game module

# --- Constants for Flappy Bird Environment ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BIRD_X = 50 # Bird's fixed X position

MODEL_FILENAME = "flappy_ai_model_manual.json"

# --- Custom Flappy Bird Environment for Q-Learning ---
class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        self.game = flappybird.Game() # Create an instance of your game

        # Define observation space and action space
        # State: (bird_y_bucket, bird_velocity_bucket, dist_to_pipe_x_bucket, pipe_gap_y_bucket)
        
        self.bird_y_bins = 20 # Discretize bird's y position
        self.bird_vel_bins = 20 # Discretize bird's vertical velocity
        self.pipe_x_bins = 20 # Discretize horizontal distance to next pipe
        self.pipe_y_bins = 20 # Discretize vertical distance from bird to pipe gap

        self.observation_space_n = (
            self.bird_y_bins *
            self.bird_vel_bins *
            self.pipe_x_bins *
            self.pipe_y_bins
        )
        
        self.action_space_n = 2 # 0: Do nothing, 1: Jump

        class ObservationSpace: # Simple mock for observation_space.n
            def __init__(self, n):
                self.n = n
        self.observation_space = ObservationSpace(self.observation_space_n)

        class ActionSpace: # Simple mock for action_space.sample()
            def __init__(self, n):
                self.n = n
            def sample(self):
                return random.randint(0, self.n - 1)
        self.action_space = ActionSpace(self.action_space_n)

    def _get_state(self):
        """
        Calculates and returns the discretized state of the game.
        """
        game_state = self.game.get_game_state_for_env() # Get cleaner state info
        bird_y = game_state['bird_y']
        bird_vel = game_state['bird_velocity_y']
        next_pipe_x = game_state['next_pipe']['x']
        next_pipe_gap_y = game_state['next_pipe']['gap_y']

        # Discretize state values
        bird_y_bucket = min(int(bird_y / (SCREEN_HEIGHT / self.bird_y_bins)), self.bird_y_bins - 1)
        
        mapped_velocity = bird_vel + 10 # Shift velocity to positive range (e.g., -7 to ~15 becomes 3 to 25)
        bird_vel_bucket = min(max(0, int(mapped_velocity / (30 / self.bird_vel_bins))), self.bird_vel_bins - 1)

        pipe_x_dist = next_pipe_x - BIRD_X
        pipe_x_bucket = min(int(pipe_x_dist / (SCREEN_WIDTH / self.pipe_x_bins)), self.pipe_x_bins - 1)

        vertical_dist_to_gap = next_pipe_gap_y - bird_y
        mapped_vertical_dist = vertical_dist_to_gap + (SCREEN_HEIGHT // 2) # Shift to positive range
        pipe_y_bucket = min(max(0, int(mapped_vertical_dist / (SCREEN_HEIGHT / self.pipe_y_bins))), self.pipe_y_bins - 1)
        
        # Combine into a single integer for the state index
        state_index = (
            bird_y_bucket * (self.bird_vel_bins * self.pipe_x_bins * self.pipe_y_bins) +
            bird_vel_bucket * (self.pipe_x_bins * self.pipe_y_bins) +
            pipe_x_bucket * (self.pipe_y_bins) +
            pipe_y_bucket
        )
        return state_index

    def reset(self):
        """Resets the game state and returns the initial observation."""
        self.game.reset() # Call the Game instance's reset method
        return self._get_state(), {} # Return state and an empty info dict

    def step(self, action):
        """
        Applies the action to the game, updates state, and returns next_state, reward, done, info.
        Action: 0 for do nothing, 1 for jump
        """
        # Call the game's internal update method for one frame
        game_over, reward = self.game.update_game_state_one_frame(action)
        
        next_state = self._get_state()
        done = game_over
        info = {}

        return next_state, reward, done, info

    def render(self):
        """Renders the current state of the game."""
        self.game.draw() # Use the game's draw method

    def close(self):
        """Cleans up the environment."""
        pygame.quit()

# --- Q-Learning Training Loop (Manual Implementation) ---
def train(load_model, episodes=50):
    if __name__ == "__main__":
        env = FlappyBirdEnv()

        # Define Neural Network architecture
        input_size = env.observation_space.n
        output_size = env.action_space.n # 2 actions: do nothing, jump

        neuralnetwork.clear_network() 
        # Create the network: input_size, output_size, hidden_layers
        # Example: one hidden layer with 128 neurons, ReLU activation
        if load_model:
            filename = "./NeuralNetworks/" + MODEL_FILENAME
            neuralnetwork.load_network(filename)
            print(f"Loaded model NeuralNetworks/{MODEL_FILENAME} for training.")
        else:
            neuralnetwork.create_network(input_size, output_size, [[128, "relu", 1]])
            print("Created new neural network for training.")

        # Training parameters
        EPISODES = episodes
        LEARNING_RATE = 0.001
        GAMMA = 0.99 
        INITIAL_EPSILON = 1.0
        MIN_EPSILON = 0.01
        EPSILON_DECAY_RATE = 0.9995 
        CLIP_GRADIENT_VALUE = 1.0 # Helps prevent exploding gradients

        epsilon = INITIAL_EPSILON

        print("Starting manual Q-Learning training...")
        start_time = time.time()

        for episode in range(EPISODES):
            state, info = env.reset() # Reset the environment for a new episode
            done = False
            current_episode_total_reward = 0

            while not done:
                # 1. Convert state to one-hot vector for NN input
                state_vec = neuralnetwork._to_one_hot(state, env.observation_space.n)

                # 2. Get Q-values from the network
                q_values = neuralnetwork.net.forward(state_vec.reshape(1, -1))

                # 3. Epsilon-greedy action selection
                if np.random.rand() < epsilon:
                    action = env.action_space.sample() # Explore: random action
                else:
                    action = np.argmax(q_values) # Exploit: best action from Q-values

                # 4. Take action in environment and get next state, reward, done
                next_state, reward, done, info = env.step(action)
                current_episode_total_reward += reward
                
                # 5. Convert next state to one-hot vector
                next_state_vec = neuralnetwork._to_one_hot(next_state, env.observation_space.n)
                
                # 6. Get Q-values for the next state
                next_q = neuralnetwork.net.forward(next_state_vec.reshape(1, -1))

                # 7. Calculate the target Q-value
                target_q = q_values.copy() # Start with current Q-values
                if done:
                    # If episode ends, target Q is just the immediate reward
                    target_q[0, action] = reward 
                else:
                    # Bellman equation: Q(s,a) = R + gamma * max(Q(s',a'))
                    target_q[0, action] = reward + GAMMA * np.max(next_q)

                # 8. Perform backward pass to update network weights
                # We want the network's output for `state` and `action` to move towards `target_q`
                neuralnetwork.net.backward(target_q, q_values, state_vec.reshape(1, -1), 
                                        LEARNING_RATE, loss_type="mse", clip_value=CLIP_GRADIENT_VALUE)
                
                # 9. Update current state
                state = next_state
            
            # Epsilon-Decay
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY_RATE)

            print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {current_episode_total_reward}, Epsilon: {epsilon:.4f}")
            if episode % 10 == 0:
                print(f"Q-values: {q_values}")
            
            # Autosave logic
            if MODEL_FILENAME and (episode + 1) % 1000 == 0: # Save every 1000 episodes
                filename = "./NeuralNetworks/" + MODEL_FILENAME
                neuralnetwork.save_network(filename)
                print(f"Network saved at episode {episode + 1}")

        print("Training complete.")
        filename = "./NeuralNetworks/" + MODEL_FILENAME
        neuralnetwork.save_network(filename) # Save final model
        print(f"Final model saved to NeuralNetworks/{MODEL_FILENAME}")

        end_time = time.time()
        training_duration_seconds = end_time - start_time
        hours = int(training_duration_seconds // 3600)
        minutes = int((training_duration_seconds % 3600) // 60)
        print(f"Training finished in {hours} hours and {minutes} minutes.")
        env.close()

def test(episodes=5):
    # --- Optional: Load and Test Trained Agent ---
    # After training, you can uncomment and run this part to see the agent play.
    neuralnetwork.clear_network()
    filename = "./NeuralNetworks/" + MODEL_FILENAME
    neuralnetwork.load_network(filename)
    print(f"\nLoaded model NeuralNetworks/{MODEL_FILENAME} for testing.")
    test_env = FlappyBirdEnv() # Create a new environment for testing

    for _ in range(episodes): # Run 5 test games
        state, info = test_env.reset()
        done = False
        test_score = 0
        while not done:
            state_vec = neuralnetwork._to_one_hot(state, test_env.observation_space.n)
            q_values = neuralnetwork.net.forward(state_vec.reshape(1, -1))
            action = np.argmax(q_values) # Always choose best action (no exploration)

            state, reward, done, info = test_env.step(action)
            if reward == 10: # Assuming 10 is the pipe passing reward
                test_score += 1

            test_env.render()
            # pygame.time.wait(10) # Small delay for visualization

            if done:
                print(f"Test Game Over! Score: {test_score}")
                pygame.time.wait(500) # Wait a bit before next game

    test_env.close()

command = input("Command: ")
if command == "train":
    load = input("Load model (y/n): ")
    if load.lower() == 'y':
        load_model = True
    else:
        load_model = False
    episodes = int(input("Number of episodes (default 50): "))
    train(load_model, episodes)
elif command == "test":
    episodes = int(input("Number of test episodes (default 5): "))
    test(episodes)