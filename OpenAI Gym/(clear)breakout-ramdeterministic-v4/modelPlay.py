# 학습모델 play. random action과 학습model 비교.

import gym
import pylab
import numpy as np
import gym.wrappers as wrappers
from doubleDQN2 import DoubleDQNAgent


EPISODES = 5 # 처음은 random으로 수행, 나중에는 학습model로 수행
global_step = 0

def change_action(action):
    if action == 0:
        return 0
    elif action == 1:
        return 2
    elif action == 2:
        return 3
    elif action == 3:
        return 3

if __name__ == "__main__":
    env = gym.make('Breakout-ramDeterministic-v4')
    env = wrappers.Monitor(env,"./results",force = True)
    state_size = 128
    action_size = 3

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
        life = 5
        env.reset()
        for i in range(5):
            env.step(1) # 시작 action.

        while not done:
            action = env.action_space.sample()
            _, reward, done, info = env.step(change_action(action))
            score += reward

            if done:
                if score > 0:
                    random_success_cnt += 1
                print("episode:", e, "  score:", score)
                scores.append(score)
                episodes.append(e)
                break

            if life != info['ale.lives']:
                life = info['ale.lives']
                for i in range(5):
                    state, _, _, _ = env.step(1)
                state = np.reshape(state, [1, 128])

    # 학습모델 play
    for e in range(EPISODES,EPISODES*2):
        done = False
        life = 5
        score = 0
        state = env.reset()
        for i in range(5):
            state, _, _, _ = env.step(1) # 시작 action.
        state = np.reshape(state,[1,128])

        while not done:
            global_step += 1
            if agent.render:
                env.render()

            # 현재 s에서 a취해 s`, r, done 정보 얻기.
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(change_action(action))
            score += reward
            state = next_state
            state = np.reshape(state, [1, 128])

            if done:
                if score > 0 :
                    model_success_cnt += 1
                print("episode:", e, "  score:", score)
                scores.append(score)
                episodes.append(e)
                if e % 5 == 0:
                    pylab.plot(episodes, scores)
                    pylab.savefig("./play_score.png")
                break

            if life != info['ale.lives']:
                life = info['ale.lives']
                for i in range(5):
                    state, _, _, _ = env.step(1)
                state = np.reshape(state, [1, 128])
    env.close()
    print("random : {}/{} success. rate : {}".format(random_success_cnt,EPISODES,random_success_cnt/EPISODES))
    print("model : {}/{} success. rate : {}".format(model_success_cnt,EPISODES,model_success_cnt/EPISODES))