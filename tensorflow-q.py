import numpy as np
import gym
import os
import random
import json
from collections import deque
import tensorflow as tf
import time

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# --- GPU Setup ---
# Remove TPU logic, prefer GPU if available, else CPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs found: {[d.name for d in physical_devices]}")
    except Exception as e:
        print(f"Could not set GPU memory growth: {e}")
    strategy = tf.distribute.MirroredStrategy()
    print("Using MirroredStrategy for GPU.")
else:
    print("No GPU found, running on CPU.")
    strategy = tf.distribute.get_strategy()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow is using:", tf.test.gpu_device_name())

# --- Q-Network Agent ---
class QNetworkAgent:
    def __init__(self, state_size, action_size, hidden_layers=None, learning_rate=0.001, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers or [128]
        # Model must be created within the strategy scope
        with strategy.scope():
            self.model = self._build_model()
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.995

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.state_size,)))
        for units in self.hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=64):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])
        q_targets = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        for i in range(len(minibatch)):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.amax(q_next[i])
            q_targets[i][actions[i]] = target
        self.model.fit(states, q_targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        self.model.save(filename)
        # Save epsilon and other params
        params = {
            'epsilon': self.epsilon,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate
        }
        with open(filename + '.meta.json', 'w') as f:
            json.dump(params, f)

    def load(self, filename):
        self.model = tf.keras.models.load_model(filename)
        meta_path = filename + '.meta.json'
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                params = json.load(f)
                self.epsilon = params.get('epsilon', 1.0)
                self.hidden_layers = params.get('hidden_layers', [128])
                self.learning_rate = params.get('learning_rate', 0.001)

# --- Helper functions ---
def one_hot_state(state, state_space):
    arr = np.zeros(state_space)
    arr[state] = 1.0
    return arr

# --- Flappy Bird Environment Wrapper ---
class FlappyBirdEnv:
    def __init__(self):
        from flappybird import Game
        self.game = Game()
        self.action_space = 2
        self.state_size = 4

    def reset(self):
        self.game.reset()
        return self._get_state()

    def step(self, action):
        done, reward = self.game.update_game_state_one_frame(action)
        state = self._get_state()
        return state, reward, done, {}

    def render(self):
        self.game.draw()

    def _get_state(self):
        info = self.game.get_game_state_for_env()
        bird_y = info['bird_y'] / 600.0  # normalize
        bird_vy = info['bird_velocity_y'] / 10.0
        next_pipe_x = (info['next_pipe']['x'] - 50) / 800.0  # bird x is 50
        next_pipe_gap_y = info['next_pipe']['gap_y'] / 600.0
        return np.array([bird_y, bird_vy, next_pipe_x, next_pipe_gap_y], dtype=np.float32)

# --- Main Command Loop ---

env = FlappyBirdEnv()
state_size = env.state_size
action_size = env.action_space
agent = None

print("Flappy Bird TensorFlow Q-Learning Agent. Type 'help' for commands.")

while True:
    command = input("Command: ").strip().lower()
    if command == 'exit':
        break
    elif command == 'create':
        agent = QNetworkAgent(state_size, action_size, hidden_layers=[128, 128], learning_rate=0.001)
        print("Agent created.")
    elif command == 'clear':
        agent = None
        print("Agent cleared.")
    elif command == 'show':
        if agent is not None:
            print(f"Q-Network: input {agent.state_size}, output {agent.action_size}, hidden {agent.hidden_layers}")
            print(f"Epsilon: {agent.epsilon:.3f}")
        else:
            print("No agent created.")
    elif command == 'save':
        if agent is None:
            print("No agent to save.")
            continue
        filename = input("Filename: ")
        filename = os.path.join("NeuralNetworks", filename + ".keras")
        agent.save(filename)
        print(f"Agent saved to {filename}.")
    elif command == 'load':
        filename = input("Filename: ")
        filename = os.path.join("NeuralNetworks", filename + ".keras")
        if not os.path.exists(filename):
            print("File not found.")
            continue
        agent = QNetworkAgent(state_size, action_size)
        agent.load(filename)
        print(f"Agent loaded from {filename}.")
    elif command == 'train':
        if agent is None:
            print("No agent created.")
            continue
        episodes = int(input("Episodes: "))
        max_steps = int(input("Max steps per episode: "))
        batch_size = int(input("Batch size: "))
        save_interval = int(input("Save interval (0 to disable): "))
        if save_interval > 0:
            filename = input("Filename for saving model: ")
            filename = os.path.join("NeuralNetworks", filename + ".keras")
        else:
            filename = None
        agent.epsilon = 1.0 
        for ep in range(1, episodes+1):
            state = env.reset()
            total_reward = 0
            for step in range(max_steps):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            agent.replay(batch_size)
            print(f"Episode {ep}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}, Steps: {step+1}, score: {env.game.score}")
            if save_interval > 0 and filename and ep % save_interval == 0:
                agent.save(filename)
                print(f"Agent saved at episode {ep}")
        print("Training complete.")

    elif command == 'test':
        if agent is None:
            print("No agent created.")
            continue
        episodes = int(input("Test episodes: "))
        render = input("Render? (y/n): ").lower() == 'y'
        test_epsilon = float(input("Test epsilon (0 for greedy, e.g. 0.05 for some exploration): "))
        total_rewards = []
        original_epsilon = agent.epsilon
        agent.epsilon = test_epsilon
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            for _ in range(1000):
                if render:
                    env.render()
                action = agent.act(state)  # Use epsilon-greedy policy
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward
                if done:
                    break
            total_rewards.append(total_reward)
            print(f"Test Episode {ep+1}: Reward = {total_reward}, Score = {env.game.score}")
        avg_reward = np.mean(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
        agent.epsilon = original_epsilon
    elif command == 'train_pop':
        num_agents = int(input("Number of agents in population (e.g. 4): "))
        episodes = int(input("Episodes per agent (e.g. 500): "))
        max_steps = int(input("Max steps per episode (e.g. 1000): "))
        batch_size = int(input("Batch size (e.g. 64): "))
        save_interval = int(input("Save interval (0 to disable, e.g. 100): "))
        if save_interval > 0:
            filename_base = input("Filename base for saving models: ")
        else:
            filename_base = None
        # Create population of agents
        agents = [QNetworkAgent(state_size, action_size, hidden_layers=[128, 128], learning_rate=0.001, epsilon_min=0.01) for _ in range(num_agents)]
        envs = [FlappyBirdEnv() for _ in range(num_agents)]
        for idx, agent in enumerate(agents):
            agent.epsilon = 1.0
        start_time = time.time()  # Start timing
        for ep in range(1, episodes+1):
            for i, agent in enumerate(agents):
                state = envs[i].reset()
                total_reward = 0
                for step in range(max_steps):
                    action = agent.act(state)
                    next_state, reward, done, _ = envs[i].step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    if done:
                        break
                agent.replay(batch_size)
                print(f"Agent {i+1}/{num_agents} | Episode {ep}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}, Steps: {step+1}, score: {envs[i].game.score}, q-values: {agent.model.predict(state[np.newaxis, :], verbose=0)[0]}")
                if save_interval > 0 and filename_base and ep % save_interval == 0:
                    filename = os.path.join("NeuralNetworks", f"{filename_base}_agent{i+1}_ep{ep}.keras")
                    agent.save(filename)
                    print(f"Agent {i+1} saved at episode {ep}")
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        print(f"Population training complete. Time elapsed: {hours}h {minutes}m")
    elif command == 'help':
        print("Commands: create, clear, show, save, load, train, train_population, test, exit")
    else:
        print("Unknown command.")
