# ddqn keras구현모델.
# cartpole ddqn 참고 : https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/2-double-dqn/cartpole_ddqn.py

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
* agent에서 train할 때는 float32로 다시 변환함.

2. 연속 3state단위로 학습(history).(방향성 정보 위해서)
history shape = (1,40,32,3) => # 원래 history shape은 (40,32,3)이지만 keras cnn layer의
                               # input으로 받으려면 앞에 1차원 더 넣어주어야함.
초기값은 메모리에 3 시작 state를 넣음.
---
while()
    next_state를 (40,32,1)로 reshape
    next_history = next_state에 history의 마지막 state부분을 제외하고 붙이기.
---
3. layer는 CNN사용.
'''

EPISODES = 1000000
global_step = 0

def pre_processing(state):
    processed_state = np.uint8(rgb2gray(state) * 255)
    return np.reshape([processed_state],(1,40,32,1))

if __name__ == "__main__":
    env = gym.make('CubeCrash-v0')
    state_size = (40,32,3) # (40,32) state를 3개 history씩 입력.
    action_size = env.action_space.n    #3

    agent = DoubleDQNAgent(state_size, action_size)

    scores, episodes, avg_q_max_record = [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        epi_step = 0
        agent.avg_q_max = 0
        state = env.reset()
        state = pre_processing(state) # state 전처리
        # history 초기값
        history = np.stack((state,state,state),axis=2)
        history = np.reshape([history], (1,40,32,3))

        while not done:
            global_step += 1
            epi_step += 1
            if agent.render:
                env.render()

            # 현재 s에서 a취해 s`, r, done 정보 얻기.
            action = agent.get_action(history)
            next_state, reward, done, info = env.step(action)
            next_state = pre_processing(next_state) # next_state 전처리 shape:(1,40,32,1)
            next_history = np.append(next_state,history[:,:,:,:2],axis=3) # 마지막 state없애고 새 state history에 넣기
            #reward = reward if not done else -100 # CubeCrash-v0는 reward에 weight 이미 주어져있음.

            agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0]) # 학습 잘 되는지 확인 변수.

            # replay memory에 <s, a, r, s',done> 저장.
            agent.append_sample(history, action, reward, next_history, done)
            # 정보들 기반해 학습.
            agent.train_model()
            score += reward
            history = next_history

            # 일정 주기마다 타겟모델 = 학습모델 동기화
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            if done:
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory), "  epsilon:", agent.epsilon)
                scores.append(score)
                episodes.append(e)
                avg_q_max_record.append(agent.avg_q_max / float(epi_step))
                if e % 100 == 0:
                    pylab.figure(1) # score 그래프
                    pylab.plot(episodes, scores)
                    pylab.savefig("./ddqn.png")
                    pylab.figure(2) # avg_q_max 그래프
                    pylab.plot(episodes, avg_q_max_record)
                    pylab.savefig("./avg_q_max.png")

        # 주기마다 모델 저장.
        if e % 100 == 0:
            print("{} episode까지 학습모델 저장".format(e))
            agent.model.save_weights("./ddqn.h5")