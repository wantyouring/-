# 학습모델 play. random action과 학습model 비교.

import gym
import pylab
import numpy as np
from skimage.color import rgb2gray
from doubleDQN import DoubleDQNAgent

EPISODES = 500 # 처음은 random으로 수행, 나중에는 학습model로 수행
global_step = 0

def pre_processing(state):
    processed_state = np.uint8(rgb2gray(state) * 255)
    return np.reshape([processed_state], (1, 40, 32, 1))

if __name__ == "__main__":
    env = gym.make('CubeCrash-v0')
    state_size = (40, 32, 3)  # (40,32) state를 3개 history씩 입력.
    action_size = env.action_space.n  # 3

    agent = DoubleDQNAgent(state_size, action_size)
    agent.load_model()
    agent.epsilon = -1 # Q value에 의해서만 움직이게끔.
    agent.render = True

    scores, episodes = [], []
    random_success_cnt = 0
    model_success_cnt = 0
    # 랜덤액션 진행시
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()

        while not done:
            _ , reward, done, _ = env.step(env.action_space.sample())
            score += reward

            if done:
                if score > 0:
                    random_success_cnt += 1
                print("episode:", e, "  score:", score)
                scores.append(score)
                episodes.append(e)

    for e in range(EPISODES,EPISODES*2):
        done = False
        score = 0
        state = env.reset()
        # state = np.reshape(state, [1, state_size])
        state = pre_processing(state)  # state 전처리
        # history 초기값
        history = np.stack((state, state, state), axis=2)
        history = np.reshape([history], (1, 40, 32, 3))
        # print(np.reshape(history,(40,32,3)))

        while not done:
            global_step += 1
            if agent.render:
                env.render()

            # 현재 s에서 a취해 s`, r, done 정보 얻기.
            action = agent.get_action(history)
            next_state, reward, done, _ = env.step(action)
            next_state = pre_processing(next_state)  # next_state 전처리 shape:(40,32,1)
            next_history = np.append(next_state, history[:, :, :, :2], axis=3)  # 마지막 state없애고 새 state history에 넣기

            score += reward
            history = next_history

            if done:
                if score > 0 :
                    model_success_cnt += 1
                print("episode:", e, "  score:", score)
                scores.append(score)
                episodes.append(e)
                if e % 20 == 0:
                    pylab.plot(episodes, scores)
                    pylab.savefig("./play_score.png")
    print("random : {}/{} success. rate : {}".format(random_success_cnt,EPISODES,random_success_cnt/EPISODES))
    print("model : {}/{} success. rate : {}".format(model_success_cnt,EPISODES,model_success_cnt/EPISODES))