# ddqn keras구현모델.
# cartpole ddqn 참고 : https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/2-double-dqn/cartpole_ddqn.py

import gym
import pylab
import numpy as np
import sys
from skimage.color import rgb2gray
from doubleDQN import DoubleDQNAgent

'''
1. action => 정지, 1: 시작, 2: 오른쪽 이동, 3: 왼쪽 이동
  ㄴ시작할 때 1action 쓰고, 그 후로는 action 0,2,3만 사용.
  action space를 3으로 주고 get action에서 0,1,2 =>0,2,3으로 치환해주기.
2. 연속 3state단위로 학습(history).(방향성 정보 위해서)
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
    state_size = 128*3 # (128) state를 3개 history씩 입력.
    action_size = 3 # start action 제외.

    agent = DoubleDQNAgent(state_size, action_size)
    #agent.load_model() #@@@@@@@@모델로드
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
        # history 초기값
        #history = np.stack((state,state,state),axis=1)
        history_t = np.hstack((state,state,state)) # (6,)
        history = np.reshape(history_t,[1,128*3]) # (1,6) => model의 메소드에는 이 형식으로 넣어야함.
        #history = np.reshape([history], (1,128,3))
        while not done:
            global_step += 1
            epi_step += 1
            if agent.render:
                env.render()

            # 현재 s에서 a취해 s`, r, done 정보 얻기.
            action = agent.get_action(history)
            next_state, reward, done, info = env.step(change_action(action)) # step취할 때 env의 action맞게 치환.
            next_history_t = np.append(next_state,history_t[:256])
            next_history = np.reshape(next_history_t,[1,128*3])
            #next_history = np.append(next_state,history[:,:256],axis=1)
            #next_history = np.append(next_state,history[:,:,:2],axis=2) # 마지막 state없애고 새 state history에 넣기
            #reward = reward if not done else -100 # reward 설정. 보류.

            agent.avg_q_max += np.amax(agent.model.predict(history)[0]) # 학습 잘 되는지 확인 변수. qvalue 최댓값.

            # replay memory에 <s, a, r, s',done> 저장.
            if life != info['ale.lives']:
                agent.append_sample(history, action, reward - 100, next_history, done)
            else:
                agent.append_sample(history, action, reward, next_history, done)
            #agent.append_sample(history, action, reward, next_history, done)
            # 정보들 기반해 학습.
            agent.train_model()
            score += reward
            history = next_history
            history_t = next_history_t

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
                history_t = np.hstack((state, state, state))  # (6,)
                history = np.reshape(history_t, [1, 128 * 3])

        # 주기마다 모델 저장.
        if e % 100 == 0:
            print("{} episode까지 학습모델 저장".format(e))
            agent.model.save_weights("./ddqn.h5")