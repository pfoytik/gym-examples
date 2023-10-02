import numpy as np
import gymnasium as gym
import matrix_mdp
import random
import math
from collections import namedtuple, deque
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from tqdm.notebook import trange

def initialize_q_table(state_space, action_space):
    q_table = np.zeros((state_space, action_space))
    return q_table

def epsilon_greedy_policy(Qtable, state, epsilon):
    print(state)
    print(Qtable)
    if random.uniform(0, 1) < epsilon:        
        return random.randint(0, action_space - 1)
    else:
        return np.argmax(Qtable[state])

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in range(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        state, obj = env.reset()
        step = 0
        done = False

    for step in range(max_steps):
        action = epsilon_greedy_policy(Qtable, state, epsilon)
        new_state, reward, done, info, _ = env.step(action)
        Qtable[state, action] = Qtable[state, action] + learning_rate * (reward + gamm * np.max(Qtable[new_state, :]) - Qtable[state, action])

        state = new_state
        if done == True:
            break
    return Qtable

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    episode_rewards = []
    for episode in range(n_eval_episodes):
        if seed:
            state, obj = env.reset(seed=seed[episode])
        else:
            state, obj = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action = np.argmax(Q[state, :])
            new_state, reward, done, info, _ = env.step(action)
            total_rewards_ep += reward
            state = new_state
            if done == True:
                break
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward

## Example probabilities
#p_0 = np.array([0.1, 0.9])
#p = np.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
#p = np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]])

# Generate probabilities and reward matrices
n_states = range(3)
n_actions = range(2)
#p_0, p = generate_probabilities(n_states, n_actions)
#r = generate_rewards(n_states, n_actions)

## Probability(p, state, action)

p_0 = np.array([0.1, 0.8, 0.1])

## Default wikipedia MDP example
p = np.array([[[0.5, 0.0], [0.7, 0.0], [0.4, 0.3]],
             [[0.5, 1.0], [0.1, 0.95], [0.0, 0.3]],
             [[0.0, 0.0], [0.2, 0.05], [0.6, 0.4]]])

### Default wikipedia MDP example 106.55
r1 = np.array([[[0.0, 0.0], [5, 0.0], [0.0, -1]],
             [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
             [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])

### HIgher than default avg = 155.25
r2 = np.array([[[-1.0, 0.0], [5, 0.0], [0.0, -1]],
             [[0.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],
             [[0.0, 5.0], [0.0, 0.0], [-1.0, 0.0]]])

### HIgher than default avg = 184.04
r = np.array([[[-1.0, 0.0], [5, 0.0], [0.0, 5]],
             [[0.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],
             [[0.0, 5.0], [0.0, 0.0], [-1.0, 0.0]]])

# Save probabilities and reward matrices
np.save("data/AICL1_MDP_P_0.npy", p_0)
np.save("data/AICL1_MDP_P_Sprime_S_A.npy", p)
np.save("data/AICL1_MDP_R_Sprime_S_A.npy", r)

# Load probabilities and reward matrices
r = np.load("data/AICL1_MDP_R_Sprime_S_A.npy")
p_0 = np.load("data/AICL1_MDP_P_0.npy")
p = np.load("data/AICL1_MDP_P_Sprime_S_A.npy")


'''
## Print matrices to check structure
print(p)
print("###################")
print(r)
print("###################")
print(p_0)
print("###################")

for s in n_states:
    for a in n_actions:        
        #print(s, a)
        #print((p[s, a, 0] + p[s, a, 1]),s, a, p[s, a, 0], p[s, a, 1])        
        print(p[:,s,a].sum())

print("###################")
'''

state_space = len(n_states)
action_space = len(n_actions)

# Create environment
env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=p_0, p=p, r=r)

Qtable = initialize_q_table(state_space, action_space)

n_training_episodes = 1000
learning_rate = 0.7
n_eval_episodes = 100

env_id = 'matrix_mdp/MatrixMDP-v0'
max_steps = 99
gamm = 0.95
eval_seed = []

max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.005

Qtable = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable)

# Evaluate our Agent
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

print("#####################")
print(Qtable)

### Old tests
'''
obs, obj = env.reset()

for i in range(0,10):
    if obs == 0:
        obs, rew, d1, d2, obj = env.step(1)
    elif obs == 1:
        obs, rew, d1, d2, obj = env.step(0)
    elif obs == 2:
        obs, rew, d1, d2, obj = env.step(1)
    else:
        obs, rew, d1, d2, obj = env.step(i%2)
    print(obs, rew, d1, d2, obj)
'''
