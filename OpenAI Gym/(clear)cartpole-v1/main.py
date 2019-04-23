# ddqn keras구현모델.
# https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/2-double-dqn/cartpole_ddqn.py

import gym
import pylab
import numpy as np
import sys
from doubleDQN import DoubleDQNAgent

EPISODES = 300

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0] #4
    action_size = env.action_space.n    #2

    agent = DoubleDQNAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # 현재 s에서 a취해 s`, r, done 정보 얻기.
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            # replay memory에 <s, a, r, s',done> 저장.
            agent.append_sample(state, action, reward, next_state, done)
            # 정보들 기반해 학습.
            agent.train_model()
            score += reward
            state = next_state

            if done:
                # 한 episode마다 타겟모델 = 학습모델 동기화.
                agent.update_target_model()

                score = score if score == 500 else score + 100
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory), "  epsilon:", agent.epsilon)
                scores.append(score)
                episodes.append(e)
                if e % 20 == 0:
                    pylab.plot(episodes, scores)
                    pylab.savefig("./cartpole_ddqn.png")

                # 최종 10 episode 평균 > 490면 학습 중단.
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # 주기마다 모델 저장.
        if e % 50 == 0:
            agent.model.save_weights("./cartpole_ddqn.h5")