#frozen lake에서 랜덤으로 진행했을 때 성공률 계산해보기.
# 약 0.01정도 나옴

import gym
env = gym.make('FrozenLake-v0')
epi_num = 10000
success = 0
for episode in range(epi_num):
    done = False
    env.reset()
    while not done:
        #env.render()
        action = env.action_space.sample()
        _ , reward, done, _ = env.step(action)
        #print("reward : {}".format(reward))
        if done and reward == 1:
            success += 1
print(success/epi_num)
env.close()