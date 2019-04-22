# Taxi-v2 환경을 학습해보자
# 큰 discrete observation space에서의 dqn적용.

import numpy as np
import tensorflow as tf
import random
from collections import deque
import dqn
import pylab
import gym
from typing import List

env = gym.make('Taxi-v2')
save_file = './model.ckpt'

# action space n : Discrete(8)
# observation space shape : Discrete(500)
# Observations:
#     There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is the taxi), and 4 destination locations.

INPUT_SIZE = env.observation_space.n #one hot 인코딩으로 전처리.
OUTPUT_SIZE = env.action_space.n

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 2000 #50000   #5000 memory error.
BATCH_SIZE = 32 #32
TARGET_UPDATE_FREQUENCY = 5
MAX_EPISODES = 10000
LOAD_MODEL = False

def one_hot(x):
    return np.identity(500)[x:x+1]

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
        for episode in range(MAX_EPISODES):
            e = 1. / ((episode / 100) + 1) # episode/10
            done = False
            step_count = 0
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
                    action = np.argmax(mainDQN.predict(one_hot(state)))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)
                rSum += reward

                # Save the experience to our buffer
                replay_buffer.append((one_hot(state), action, reward, one_hot(next_state), done))

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                # 일정 주기로 타겟 업데이트
                if step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)

                state = next_state
                step_count += 1
            print("{} Episode total reward : {}".format(episode,rSum))
            graph_x.append(episode)
            graph_y.append(rSum)
            if episode % 50 == 0 and episode != 0:
                pylab.plot(graph_x, graph_y)
                pylab.savefig("./dqn_graph.png")
                #print("Episode: {} 총 reward : {}".format(episode,rSum))
            if episode % 50 == 0 and episode != 0:
                saver.save(sess, save_file)  # 50epi 주기마다 모델 save
                print("Episode: {} 모델 저장 완료".format(episode))

if __name__ == "__main__":
    main()