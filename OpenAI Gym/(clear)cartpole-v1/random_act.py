# acrobot 랜덤액션 실행 step 평균 계산

import gym
import pylab

env = gym.make('CartPole-v1')
epi_num = 500
rSum = 0
graph_x, graph_y = [],[]
for episode in range(epi_num):
    done = False
    env.reset()
    step_count = 0
    while not done :
        #env.render()
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        rSum += reward
        step_count += 1

    graph_x.append(episode)
    graph_y.append(step_count)
    if episode % 50 == 0 and episode != 0:
        pylab.plot(graph_x, graph_y)
        pylab.savefig("./random_graph.png")
print("random act end")
env.close()