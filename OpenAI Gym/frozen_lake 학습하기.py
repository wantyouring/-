# frozen_lake 학습해 성공률 계산해보기
# cart pole에 적용된 DQN 먼저 공부해보고 frozen lake에 적용하자

import gym
env = gym.make('FrozenLake-v0')
epi_num = 10000
success = 0
