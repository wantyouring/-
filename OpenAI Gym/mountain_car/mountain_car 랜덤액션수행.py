# mountain_car 환경 랜덤액션 수행
'''
mountain_car 환경 정보
action space n : Discrete(3)
observation space shape : Box(2,)
sample action : 1
sample reward : -1.0

reward는 항상 -1 return함.
goal이면 +100 reward를 주자.
'''

import gym

env = gym.make('MountainCar-v0')
epi_num = 1
rSum = 0

env.reset()
space, reward, done, _ = env.step(env.action_space.sample())

#print("observation space n : {}".format(env.observation_space.n))
print("action space n : {}".format(env.action_space))
print("observation space shape : {}".format(env.observation_space))
print("sample action : {}".format(env.action_space.sample()))
print("sample reward : {}".format(reward))

for episode in range(epi_num):
    done = False
    env.reset()
    #while not done :
    for step in range(100):
        env.render()
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        print("{} step`s reward : {}".format(step,reward))
        rSum += reward
print("random action average reward : {}".format(rSum/epi_num))
env.close()