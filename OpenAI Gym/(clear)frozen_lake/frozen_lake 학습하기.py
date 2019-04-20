# frozen_lake 학습해 성공률 계산해보기
# (아직 keras 잘 안되므로 김성훈교수님 2015 dqn 사용)keras로 frozen lake에 적용해보자.
# 5000 episode까지 학습한 성공률 0.6674.
# 학습 성공.
# 모델 저장 및 불러오기 추가해보자. (추가 완료)
# keras로도 바꿔보자.
import numpy as np
import tensorflow as tf
import random
from collections import deque
import dqn
import pylab
import gym
from typing import List

env = gym.make('FrozenLake-v0')
save_file = './model.ckpt'
INPUT_SIZE = env.observation_space.n    #one hot 인코딩해서 input size 16가능. one hot 인코딩 안하고 학습하니 학습이 아예 안된다!
OUTPUT_SIZE = env.action_space.n

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
MAX_EPISODES = 5000
LOAD_MODEL = True

def one_hot(x):
    return np.identity(16)[x:x+1]

def replay_train(mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list) -> float:
    """Trains `mainDQN` with target Q values given by `targetDQN`
    Args:
        mainDQN (dqn.DQN): Main DQN that will be trained
        targetDQN (dqn.DQN): Target DQN that will predict Q_target
        train_batch (list): Minibatch of replay memory
            Each element is (s, a, r, s', done)
            [(state, action, reward, next_state, done), ...]
    Returns:
        float: After updating `mainDQN`, it returns a `loss`
    """
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
    """Creates TF operations that copy weights from `src_scope` to `dest_scope`
    Args:
        dest_scope_name (str): Destination weights (copy to)
        src_scope_name (str): Source weight (copy from)
    Returns:
        List[tf.Operation]: Update operations are created and returned
    """
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def bot_play(mainDQN: dqn.DQN, env: gym.Env) -> None:
    """Test runs with rendering and prints the total score
    Args:
        mainDQN (dqn.DQN): DQN agent to run a test
        env (gym.Env): Gym Environment
    """
    state = env.reset()
    reward_sum = 0

    while True:

        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward

        if done:
            print("Total score: {}".format(reward_sum))
            break

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
        success_cnt = 0

        for episode in range(MAX_EPISODES):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(one_hot(state)))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)

                if done:
                    if reward == 0: # 실패
                        reward = -10
                    else:   # 성공
                        reward = 10
                        success_cnt += 1

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

            graph_x.append(episode)
            graph_y.append(reward)
            if episode % 50 == 0 and episode != 0:
                pylab.plot(graph_x, graph_y)
                pylab.savefig("./frozen_save/frozen_lake_dqn.png")
                print("Episode: {} 까지 성공률: {}".format(episode, success_cnt / episode))
            if episode % 500 == 0 and episode != 0:
                saver.save(sess, save_file)  # 50주기마다 모델 save
                print("Episode: {} 모델 저장 완료".format(episode))


if __name__ == "__main__":
    main()