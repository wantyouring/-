# 랜덤액션 실행

import gym
import pylab

env = gym.make('Blackjack-v0')
epi_num = 50000
rSum = 0
graph_x, graph_y = [],[]
for episode in range(epi_num):
    done = False
    env.reset()
    if(episode % 100 == 0):
        rSum = 0
    while not done :
        #env.render()
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        rSum += reward

    graph_x.append(episode)
    graph_y.append(rSum)
    if episode % 50 == 0 and episode != 0:
        pylab.plot(graph_x, graph_y)
        pylab.savefig("./random_graph.png")
env.close()