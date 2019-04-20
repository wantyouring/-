# cart_pole DQN으로 학습하기. keras 사용
# https://github.com/keon/deep-q-learning/blob/master/dqn.py

# plt으로 reward 그래프 찍어보자
# (?) 타겟 모델 다시 공부하기.
# (?) reshape하는 이유??
# (?) 학습 잘 안되는 이유?

import random
import gym
import numpy as np
import pylab
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 2000
LOAD_MODEL = False

class DQNAgent:
    # 초기화자.
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    # _네이밍 : private으로 내부적으로만 쓰고 싶을 경우
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # 입력층 크기 : state 차원. 출력층 크기 : action 갯수.
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    # memory 덱에 추가
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    # exploit할지 exploration할지 epsilon에 따라 선택해 action return.
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    # memory 덱에서 batch크기만큼 랜덤으로 가져와서 학습.
    # (?) 타겟 모델 다시 공부하기.
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state) #현재 상태에 대한 모델의 큐함수
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    #모델 로드, 저장.
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0] #shape[0]로 state차원 확인. (4)
    action_size = env.action_space.n    #action space 갯수 확인(2)
    agent = DQNAgent(state_size, action_size)
    if LOAD_MODEL == True:
        agent.load("./cart_save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    graph_x, graph_y = [], []

    for e in range(EPISODES):
        total_reward = 0
        state = env.reset()
        # (?) reshape하는 이유??
        state = np.reshape(state, [1, state_size])
        for time in range(300):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10 # 종료시에는 낮은 reward주기
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done or time == 299:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                graph_x.append(e)
                graph_y.append(total_reward)
                if e % 50 == 0 :
                    pylab.plot(graph_x,graph_y,'b')
                    pylab.savefig("./cart_save/cartpole_dqn.png")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 50 == 0:
            agent.save("./cart_save/cartpole-dqn.h5")