import numpy as np
import gymnasium as gym
import matrix_mdp

# Load probabilities and reward matrices
r = np.load("data/AICL1_MDP_R_Sprime_S_A.npy")
p_0 = np.load("data/AICL1_MDP_P_0.npy")
p = np.load("data/AICL1_MDP_P_Sprime_S_A.npy")

# Create environment
env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=p_0, p=p, r=r)
