# blackjack-v0 환경을 학습해보자
# observation space가 tuple인 환경에서 dqn적용.

import numpy as np
import tensorflow as tf
import random
from collections import deque
import dqn
import pylab
import gym
from typing import List

env = gym.make('Blackjack-v0')
save_file = './model.ckpt'

# observation space shape : Tuple(Discrete(32), Discrete(11), Discrete(2))
# action space n : Discrete(2)
# tuple인 observation space를 튜플 각 원소 원핫인코딩하고 합쳐 재구성.

INPUT_SIZE = 45 #state 재구성.
OUTPUT_SIZE = env.action_space.n #2

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
BATCH_SIZE = 32 #32
TARGET_UPDATE_FREQUENCY = 5
MAX_EPISODES = 1000000
LOAD_MODEL = False

# x와 전체size로 원핫인코딩
def one_hot(x,size):
    return np.identity(size)[x:x+1]

# state 재구성. Tuple(Discrete(32), Discrete(11), Discrete(2)) => array(45 size)
def state_reform(state):
    return np.hstack([one_hot(state[0],32),one_hot(state[1],11),one_hot(state[2],2)])

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
        rSum = 0
        for episode in range(MAX_EPISODES):
            e = 1. / ((episode / 10000) + 1) # episode/100
            done = False
            step_count = 0
            if episode % 100 == True: # 100게임 단위 reward합 구하기
                rSum = 0
            state = env.reset()

            while not done:
                # 100epi마다 render
                #if episode%100 == 0 and episode != 0 :
                #    env.render()

                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state_reform(state)))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)
                rSum += reward

                # Save the experience to our buffer
                replay_buffer.append((state_reform(state), action, reward, state_reform(next_state), done))

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                # 일정 주기로 타겟 업데이트
                if step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)

                state = next_state
                step_count += 1
            #print("{} Episode total reward : {}".format(episode,rSum))
            graph_x.append(episode)
            graph_y.append(rSum)
            if episode % 1000 == 0 and episode != 0:
                pylab.plot(graph_x, graph_y)
                pylab.savefig("./dqn_graph.png")
                print("{}~{} Episode total reward : {}".format(episode-100,episode,rSum))
            if episode % 1000 == 0 and episode != 0:
                saver.save(sess, save_file)
                print("Episode: {} 모델 저장 완료".format(episode))

if __name__ == "__main__":
    main()