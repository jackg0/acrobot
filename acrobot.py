'''
Author: Jack Geissinger
Date: October 17, 2018

True Online Sarsa(lambda) implementation for Acrobot

References: [1] Reinforcement learning: An introduction by RS Sutton, AG Barto, p. 252
            [2] R Sutton, "Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding", NIPS 1996.
'''

import gym
import numpy as np
import math
from tiles3 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Initialize tile vector x and weight vector w
Size is 15552 due to there being 4 variables and 6 regions for each variable,
with 12 tiles overall, meaning 12*6^4 = 15552 (may be incorrect)
'''
x = np.zeros((15552,))
w = np.zeros((15552,))
iht = IHT(15552)

'''
Initialize constants used in algorithm.
alpha is the learning rate, gamma is the discount factor,
l (lambda) is the decay rate for the eligibility trace.
'''
alpha = 0.0166
gamma = 1
l = 0.9

'''
Initialize bounds of theta and theta dot.
'''
min_th, max_th = -np.pi, np.pi
max_speed1 = 4*np.pi
max_speed2 = 9*np.pi

'''
decode takes in a state and turns in into indices which will modify x
'''
def decode(state, action):
    th1 = state[0]
    th2 = state[1]
    th1dot = state[2]
    th2dot = state[3]

    th1_scale = 6*th1//(max_th - min_th)
    th2_scale = 6*th2//(max_th - min_th)
    th1dot_scale = 6*th1dot//(max_speed1 + max_speed1)
    th2dot_scale = 6*th2dot//(max_speed2 + max_speed2)


    indices = tiles(iht, 12, [th1_scale, th2_scale, th1dot_scale, th2dot_scale], [action])

    return indices

'''
eps-greedy chooses the action which results in the maximum weight
'''
def eps_greedy(state):
    # init an array to store optional indices for greedy selection
    options = []
    # loop through possible actions to compare them all
    for possible_action in range(3):
        # possible actions are [-1, 0, 1], we want to get indices for each
        indices = decode(state, possible_action-1)
        # add indices to our options
        options.append(indices)
    # make an array of the approximate value function values to choose from
    values = [sum(w[indices]) for indices in options]
    # make a choice, choosing the max value
    choice = values.index(max(values))
    # get the best set of indices
    bestIndices = options[choice]
    # get the best action
    choices = [0, 1, 2]
    action = choices[choice]
    return bestIndices, action

'''
Actual algorithm implementation
'''
complete = 0
env = gym.make('Acrobot-v1')
for i_episode in range(1000):
    # initialize state
    observation = env.reset()
    s1 = np.arccos(observation[0])
    s2 = np.arccos(observation[2])
    state = [s1, s1, observation[4], observation[5]]
    # take a new state and action greedily
    indices, action = eps_greedy(state)

    # initialize x for a new episode
    x = np.zeros((15552,))
    x[indices] = 1

    # initialize eligibility trace for new episode
    z = np.zeros((15552,))

    Qold = 0

    for t in range(200):
        if i_episode > 500:
            env.render()
        # Take action with new state and reward
        observation, reward, done, info = env.step(action)
        s1 = np.arccos(observation[0])
        s2 = np.arccos(observation[2])
        state = [s1, s1, observation[4], observation[5]]
        # greedily choose next action and state
        new_indices, action_p = eps_greedy(state)

        # create new x' variable
        x_p = np.zeros((15552,))
        x_p[new_indices] = 1

        # modify Q and Q'
        Q = sum(w[indices])
        Q_p = sum(w[new_indices])

        delta = reward + gamma*Q_p - Q # updated error, discounting future prediction

        # modify eligibility trace
        z = gamma*l*z + (1 - alpha*gamma*l*sum(z[indices]))*x

        # update weights we are learning
        w = w + alpha*(delta + Q - Qold)*z - alpha*(Q - Qold)*x

        # modify scalar Qold
        Qold = Q_p

        # update x and action for next timestep
        x = x_p
        action = action_p
        indices = new_indices
        if done:
            print("Finished in {} timesteps".format(t+1))
            complete += 1
            break

    print("Finished episode {}".format(i_episode+1))
print("The number of successful trials: {}".format(complete))
