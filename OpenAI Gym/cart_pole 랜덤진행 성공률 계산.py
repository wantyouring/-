# cartpole 랜덤액션 실행 reward 평균 계산
# 20정도

import gym

env = gym.make('CartPole-v1')
epi_num = 100
rSum = 0
for episode in range(epi_num):
    done = False
    env.reset()
    while not done :
        #env.render()
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        rSum += reward
print("random action average reward : {}".format(rSum/epi_num))
env.close()