
import gym
import numpy as np
import sys
import matplotlib.pyplot as plt
import pylab
from agents.DDPG import DDPG

LOAD_MODEL = True

env = gym.make('MountainCarContinuous-v0')

agent = DDPG(env)

episodes = 500
rewards = []
epi = []

if LOAD_MODEL == True :
    agent.load_model('./trained_actor.h5')

for i in range(episodes):
    state = agent.reset_episode(env)
    score = 0

    for t in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        score += reward

        if done:
            break
    epi.append(i)
    rewards.append(score)
    if i % 30 == 0 and i != 0:
        pylab.plot(epi, rewards)
        pylab.savefig("./ddpg_graph.png")
        agent.save_model('./trained_actor.h5')
    print("Episode:{}, Score: {}, Average Score: {}".format(i, score, np.mean(rewards)))