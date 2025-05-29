import gym
import numpy as np
import random
import pickle

# Hyperparameters (defaults)
default_learning_rate = 0.7
default_gamma = 0.99
default_epsilon = 1.0
default_epsilon_min = 0.1
default_epsilon_decay = 0.995
default_episodes = 1000
default_max_steps = 200

# Environment
env = gym.make('Taxi-v3', render_mode='human')
state_size = env.observation_space.n
action_size = env.action_space.n

# Q-table initialization
Q = np.zeros((state_size, action_size))

def train_agent(episodes=default_episodes, learning_rate=default_learning_rate, gamma=default_gamma, epsilon=default_epsilon, epsilon_min=default_epsilon_min, epsilon_decay=default_epsilon_decay, max_steps=default_max_steps):
    global Q
    for e in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        total_reward = 0
        for time in range(max_steps):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            best_next = np.max(Q[next_state, :])
            Q[state, action] = Q[state, action] + learning_rate * (reward + gamma * best_next - Q[state, action])
            state = next_state
            total_reward += reward
            if done:
                break
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

def test_agent(max_steps=default_max_steps):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    total_reward = 0
    for time in range(max_steps):
        action = np.argmax(Q[state, :])
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_result
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        state = next_state
        total_reward += reward
        if done:
            print(f"Test Episode: Total Reward: {total_reward}")
            break
    env.close()

def save_qtable(filename):
    with open(filename, 'wb') as f:
        pickle.dump(Q, f)
    print(f"Q-table saved to {filename}")

def load_qtable(filename):
    global Q
    with open(filename, 'rb') as f:
        Q = pickle.load(f)
    print(f"Q-table loaded from {filename}")

def clear_qtable():
    global Q
    Q = np.zeros((state_size, action_size))
    print("Q-table reset to zeros.")

def show_qtable():
    print(Q)

# Command loop
while True:
    command = input("Command: ").strip().lower()
    if command == "exit":
        break
    elif command == "train":
        episodes = input(f"Episodes (default {default_episodes}): ")
        episodes = int(episodes) if episodes else default_episodes
        learning_rate = input(f"Learning rate (default {default_learning_rate}): ")
        learning_rate = float(learning_rate) if learning_rate else default_learning_rate
        gamma = input(f"Gamma (default {default_gamma}): ")
        gamma = float(gamma) if gamma else default_gamma
        epsilon = input(f"Epsilon (default {default_epsilon}): ")
        epsilon = float(epsilon) if epsilon else default_epsilon
        epsilon_min = input(f"Epsilon min (default {default_epsilon_min}): ")
        epsilon_min = float(epsilon_min) if epsilon_min else default_epsilon_min
        epsilon_decay = input(f"Epsilon decay (default {default_epsilon_decay}): ")
        epsilon_decay = float(epsilon_decay) if epsilon_decay else default_epsilon_decay
        train_agent(episodes, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay)
    elif command == "test":
        test_agent()
    elif command == "save":
        filename = input("Filename to save Q-table: ")
        save_qtable(filename)
    elif command == "load":
        filename = input("Filename to load Q-table: ")
        load_qtable(filename)
    elif command == "clear":
        clear_qtable()
    elif command == "show":
        show_qtable()
    else:
        print("Unknown command.")