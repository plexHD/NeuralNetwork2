import numpy as np
import gym
import os
import random
import json
from collections import deque
import tensorflow as tf


# --- Q-Network Agent ---
class QNetworkAgent:
    def __init__(self, state_size, action_size, hidden_layers=None, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers or [128]
        self.model = self._build_model()
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
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

# --- Main Command Loop ---

env = gym.make('Taxi-v3')
state_space = env.observation_space.n
action_space = env.action_space.n
agent = None

print("Taxi-v3 TensorFlow Q-Learning Agent. Type 'help' for commands.")

while True:
    command = input("Command: ").strip().lower()
    if command == 'exit':
        break
    elif command == 'create':
        agent = QNetworkAgent(state_space, action_space, hidden_layers=[128], learning_rate=0.001)
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
        agent = QNetworkAgent(state_space, action_space)
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
        for ep in range(1, episodes+1):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            state_vec = one_hot_state(state, state_space)
            total_reward = 0
            for step in range(max_steps):
                action = agent.act(state_vec)
                next_state, reward, done, *_ = env.step(action)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                next_state_vec = one_hot_state(next_state, state_space)
                agent.remember(state_vec, action, reward, next_state_vec, done)
                state_vec = next_state_vec
                total_reward += reward
                if done:
                    break
            agent.replay(batch_size)
            print(f"Episode {ep}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
            if save_interval > 0 and filename and ep % save_interval == 0:
                agent.save(filename)
                print(f"Agent saved at episode {ep}")
        print("Training complete.")
    elif command == 'test':
        if agent is None:
            print("No agent created.")
            continue
        episodes = int(input("Test episodes: "))
        total_rewards = []
        for ep in range(episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            state_vec = one_hot_state(state, state_space)
            total_reward = 0
            for _ in range(200):
                action = np.argmax(agent.model.predict(state_vec[np.newaxis, :], verbose=0)[0])
                next_state, reward, done, *_ = env.step(action)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                state_vec = one_hot_state(next_state, state_space)
                total_reward += reward
                if done:
                    break
            total_rewards.append(total_reward)
            print(f"Test Episode {ep+1}: Reward = {total_reward}")
        avg_reward = np.mean(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
    elif command == 'fulltest':
        # Same as test but with more output
        if agent is None:
            print("No agent created.")
            continue
        episodes = int(input("Test episodes: "))
        for ep in range(episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            state_vec = one_hot_state(state, state_space)
            total_reward = 0
            steps = 0
            for _ in range(200):
                action = np.argmax(agent.model.predict(state_vec[np.newaxis, :], verbose=0)[0])
                next_state, reward, done, *_ = env.step(action)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                state_vec = one_hot_state(next_state, state_space)
                total_reward += reward
                steps += 1
                if done:
                    break
            print(f"Episode {ep+1}: Reward = {total_reward}, Steps = {steps}")
        print("Full test complete.")
    elif command == 'customtest':
        print("Custom test not implemented for Taxi-v3.")
    elif command == 'help':
        print("Commands: create, clear, show, save, load, train, test, fulltest, customtest, exit")
    else:
        print("Unknown command.")
