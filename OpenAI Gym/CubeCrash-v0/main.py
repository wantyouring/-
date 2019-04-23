# ddqn keras구현모델.
# https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/2-double-dqn/cartpole_ddqn.py

import gym
import pylab
import numpy as np
import sys
from skimage.color import rgb2gray
from doubleDQN import DoubleDQNAgent

'''
1. rgb2gray로 state 전처리
(40,32,3)을 rgb2gray로 (40,32)로 state차원 줄일 수 있음.
그러나 0, 0.7154, 1 총3개의 float형의 변수가 나옴.
255곱해 0,182,255 3개의 int변수로 변환 -> uint8로 메모리 save.

2. 연속 2state단위로 학습(history).(방향성 정보 위해서)
초기값은 메모리에 3 시작 state를 넣음. (1 1 1)
while()
    
    앞의 두 state history 학습.
    마지막에 새로운 state 넣음.(1 1 2)
    history 앞으로 당기기. (1 2 2)




3. layer는 CNN사용.

'''

EPISODES = 300
global_step = 0

def pre_processing(state):
    processed_state = np.uint8(rgb2gray(state) * 255)
    return processed_state

if __name__ == "__main__":
    env = gym.make('CubeCrash-v0')
    state_size = env.observation_space.shape[0] #4
    action_size = env.action_space.n    #3

    agent = DoubleDQNAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        #state = np.reshape(state, [1, state_size])
        state = pre_processing(state) # state 전처리

        while not done:
            global_step += 1
            if agent.render:
                env.render()

            # 현재 s에서 a취해 s`, r, done 정보 얻기.
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = pre_processing(next_state) # state 전처리
            #reward = reward if not done else -100 # CubeCrash-v0는 reward에 weight 이미 주어져있음.

            # replay memory에 <s, a, r, s',done> 저장.
            agent.append_sample(state, action, reward, next_state, done)
            # 정보들 기반해 학습.
            agent.train_model()
            score += reward
            state = next_state

            # 일정 주기마다 타겟모델 = 학습모델 동기화
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            if done:
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory), "  epsilon:", agent.epsilon)
                scores.append(score)
                episodes.append(e)
                if e % 20 == 0:
                    pylab.plot(episodes, scores)
                    pylab.savefig("./ddqn.png")

        # 주기마다 모델 저장.
        if e % 20 == 0:
            agent.model.save_weights("./ddqn.h5")