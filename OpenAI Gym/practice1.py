# code연습장

import gym
import pylab
import time
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

env = gym.make('Breakout-ramDeterministic-v4')
#env = gym.make('CubeCrash-v0')
f = open("log.txt","w")
np.set_printoptions(threshold=2**31-1) #배열 생략없이 표시
epi_num = 5000
rSum = 0
graph_x, graph_y = [],[]

for episode in range(epi_num):
    done = False
    env.reset()
    step_count = 0
    rSum = 0
    while not done :
        #env.render()
        action = env.action_space.sample()
        print(action)
        state, reward, done, _ = env.step(3) #0: 정지, 1: 시작, 2: 오른쪽 이동, 3: 왼쪽 이동
        # ㄴ시작할 때 1action 쓰고, 그 후로는 action 0,2,3만 사용. action space를 3으로 주고 get action에서 0,1,2 =>0,2,3으로 치환해주기.
        #processed_state = np.uint8(rgb2gray(state)*255)
        env.render()
        print(state.shape)
        print(env.action_space)
        #print(processed_state.shape)
        #f.write("state : \n{}\n".format(state))
        #f.write("action : \n{}\n".format(action))
        #f.write("processed_state : \n{}\n".format(processed_state))
        #state = processed_state
        #history = np.stack((state,state),axis=2)
        #print(history.shape)
        #f.write("processed_state : \n{}\n".format(history))
        #history = np.reshape([history],(1,40,32,2))
        #print(history.shape)
        #f.write("processed_state : \n{}\n".format(history))

        f.close()
        rSum += reward
        step_count += 1
        time.sleep(10)

    graph_x.append(episode)
    graph_y.append(step_count)
    if episode % 50 == 0 and episode != 0:
        pylab.plot(graph_x, graph_y)
        pylab.savefig("./random_graph.png")
    print("episode {} step : {}".format(episode,rSum))
#print("random action average reward : {}".format(reward / epi_num))
env.close()
