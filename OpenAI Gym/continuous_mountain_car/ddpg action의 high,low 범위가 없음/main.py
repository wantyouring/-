# continuous_mountain_car 학습해보기. DDPG 사용.
# ddpg agent참고 : https://github.com/cookbenjamin/DDPG

from ddpg import DDPG as Agent

import numpy as np
import tensorflow as tf
import random
from collections import deque
import pylab
import gym
from typing import List


env = gym.make('MountainCarContinuous-v0')
save_file = './model.ckpt'

# action space n : Box(1) => 음수는 왼쪽으로, 양수는 오른쪽으로 밈.
# observation space shape : Box(2) => (위치, 속력).
INPUT_SIZE = env.observation_space.shape[0] #2
OUTPUT_SIZE = env.action_space.shape[0] #1

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
MAX_EPISODES = 5000
LOAD_MODEL = False


def replay_train(mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list) -> float:
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states

    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)


def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def main():
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")

        saver = tf.train.Saver()
        if LOAD_MODEL == False :
            sess.run(tf.global_variables_initializer())
        elif LOAD_MODEL == True :
            saver.restore(sess, save_file)

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        graph_x, graph_y = [],[]
        all_step_count = 0
        for episode in range(MAX_EPISODES):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                if episode%100 == 0 and episode != 0 :
                    env.render()

                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)

                # 오른쪽 위의 목적지에 도달해야 하므로 높은 위치로 갈 수록 reward 크게 줘보자. => 학습 더 잘됨!
                # 추가로 적은 step에 도착할 수록 큰 reward 줘보자.
                # state => [위치,속력]. 위치 범위 (-1.2 ~ 0.6)
                if done and step_count != 199: # 성공
                    print("{} episode {} step에 도착.".format(episode,step_count))
                    #reward = 300 - step_count
                    reward = 100
                elif done : # 실패
                    reward = -1
                    print("{} episode 실패".format(episode))
                else : # 안끝난 상황
                    reward = state[0]

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                # 일정 주기로 타겟 업데이트
                if step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)

                state = next_state
                step_count += 1
                all_step_count += 1

            # 그래프는 episode별로 몇 step에 성공했는가 표기. (일찍 성공할 수록 학습 잘 된 것)
            graph_x.append(episode)
            graph_y.append(step_count)
            if episode % 50 == 0 and episode != 0:
                pylab.plot(graph_x, graph_y)
                pylab.savefig("./dqn_graph.png")
                # print("Episode: {} 까지 평균 성공 step: {}".format(episode, all_step_count / episode))
            if episode % 50 == 0 and episode != 0:
                saver.save(sess, save_file)  # 50주기마다 모델 save
                #print("Episode: {} 모델 저장 완료".format(episode))

if __name__ == "__main__":
    main()