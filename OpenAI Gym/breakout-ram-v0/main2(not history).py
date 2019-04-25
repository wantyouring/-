# ddqn keras구현모델.
# cartpole ddqn 참고 : https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/2-double-dqn/cartpole_ddqn.py

import gym
import pylab
import numpy as np
import sys
from skimage.color import rgb2gray
from doubleDQN2 import DoubleDQNAgent

'''
1. action => 정지, 1: 시작, 2: 오른쪽 이동, 3: 왼쪽 이동
  ㄴ시작할 때 1action 쓰고, 그 후로는 action 0,2,3만 사용.
  action space를 3으로 주고 get action에서 0,1,2 =>0,2,3으로 치환해주기.
3. layer는 fully connected layer 사용.
'''

EPISODES = 100000
global_step = 0

# 0,1,2 =>0,2,3 action으로 치환해주기.
def change_action(action):
    if action == 0:
        return 0
    elif action == 1:
        return 2
    elif action == 2:
        return 3

if __name__ == "__main__":
    env = gym.make('Breakout-ram-v0')
    state_size = 128
    action_size = 3 # start action 제외.

    agent = DoubleDQNAgent(state_size, action_size)

    scores, episodes, avg_q_max_record = [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        epi_step = 0
        life = 5
        agent.avg_q_max = 0
        env.reset()
        for i in range(5): # 시작 action 1을 한번만 실행해주면 자주 무시되는듯함. 정확한 이유 모름.
            state, _, _, _ = env.step(1) # 시작 action.
        state = np.reshape(state,[1,128]) # (1,128) => model의 메소드에는 이 형식으로 넣어야함.

        while not done:
            global_step += 1
            epi_step += 1
            if agent.render:
                env.render()

            # 현재 s에서 a취해 s`, r, done 정보 얻기.
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(change_action(action)) # step취할 때 env의 action맞게 치환.
            next_state = np.reshape(next_state,[1,128])

            agent.avg_q_max += np.amax(agent.model.predict(state)[0]) # 학습 잘 되는지 확인 변수. qvalue 최댓값.

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
                avg_q_max_record.append(agent.avg_q_max / float(epi_step))
                if e % 100 == 0:
                    pylab.figure(1) # score 그래프
                    pylab.plot(episodes, scores)
                    pylab.savefig("./ddqn.png")
                    pylab.figure(2) # avg_q_max 그래프
                    pylab.plot(episodes, avg_q_max_record)
                    pylab.savefig("./avg_q_max.png")
                break

            if life != info['ale.lives']: #life 하나 깎이면.
                life = info['ale.lives']
                #print(life)
                # 시작셋팅
                for i in range(5):
                    state, _, _, _ = env.step(1)
                state = np.reshape(state,[1,128])

        # 주기마다 모델 저장.
        if e % 100 == 0:
            print("{} episode까지 학습모델 저장".format(e))
            agent.model.save_weights("./ddqn.h5")