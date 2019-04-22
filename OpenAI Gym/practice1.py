# acrobot 랜덤액션 실행 step 평균 계산

import gym
import pylab
import time

env = gym.make('Taxi-v2')
epi_num = 5000
rSum = 0
graph_x, graph_y = [],[]
for episode in range(epi_num):
    done = False
    env.reset()
    step_count = 0
    rSum = 0
    while not done :
        env.render()
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        rSum += reward
        step_count += 1
        time.sleep(0.5)

    graph_x.append(episode)
    graph_y.append(step_count)
    if episode % 50 == 0 and episode != 0:
        pylab.plot(graph_x, graph_y)
        pylab.savefig("./random_graph.png")
    print("episode {} step : {}".format(episode,rSum))
#print("random action average reward : {}".format(reward / epi_num))

env.close()