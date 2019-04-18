# frozen_lake 학습해 성공률 계산해보기
# keras로 frozen lake에 적용해보자.
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import gym

env = gym.make('FrozenLake-v0')
epi_num = 1000
success = 0

