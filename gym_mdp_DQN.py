import numpy as np
import gymnasium as gym
import matrix_mdp
import random
import math
from collections import namedtuple, deque
from itertools import count
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

plt.ion()
device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_probabilities(n_states, n_actions):
    p = np.ndarray((len(n_states), len(n_states), len(n_actions)))   
    print(p.shape)
    for x in n_states:
        for i in n_states:
            r_val = random.random()
            p[x, i, 0] = r_val
            p[x, i, 1] = 1-r_val
    p_0 = np.random.randint(0,len(n_states))
    p_0 = p_0 / np.sum(p_0)
    #p = np.random.rand(n_states, n_states, n_actions)
    #p = p / np.sum(p, axis=1, keepdims=True)
    return p_0, p

def generate_rewards(n_states, n_actions):
    r = np.random.rand(len(n_states), len(n_states), len(n_actions))
    return r

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

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
#p[3,3,2]
#p = np.array([[[0.5, 0.7, 0.4], [0.0, 0.0, 0.3]], [[0.0, 0.1, 0.0], [0.0, 0.95, 0.03]], [[0.5, 0.2, 0.6], [1.0, 0.05, 0.4]]])
p = np.array([[[0.5, 0.0], [0.7, 0.0], [0.4, 0.3]],
             [[0.5, 1.0], [0.1, 0.95], [0.0, 0.3]],
             [[0.0, 0.0], [0.2, 0.05], [0.6, 0.4]]])

r = np.array([[[0.0, 0.0], [5, 0.0], [0.0, -1]],
             [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
             [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
# Save probabilities and reward matrices
np.save("data/AICL1_MDP_P_0.npy", p_0)
np.save("data/AICL1_MDP_P_Sprime_S_A.npy", p)
np.save("data/AICL1_MDP_R_Sprime_S_A.npy", r)

# Load probabilities and reward matrices
r = np.load("data/AICL1_MDP_R_Sprime_S_A.npy")
p_0 = np.load("data/AICL1_MDP_P_0.npy")
p = np.load("data/AICL1_MDP_P_Sprime_S_A.npy")

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

# Create environment
env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=p_0, p=p, r=r)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 300
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4   

n_observations = n_states

state, info = env.reset()
print(state)

policy_net = DQN(len(n_observations), len(n_actions)).to(device)
target_net = DQN(len(n_observations), len(n_actions)).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)    
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(len(n_actions))]], device=device, dtype=torch.long)
    
episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t)>=100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    criterion = nn.SmoothL1Loss()    
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor([state], device=device, dtype=torch.float)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor([observation], device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    print("Complete")
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

'''
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