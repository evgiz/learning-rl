
"""
Author: Sigve Rokenes
Date: January, 2019

This script trains a DQN actor to land on a goal
in the OpenAI LunarLander-v2 gym environment.

The algorithm converges after about 1500 episodes, and achieved
an average score of 266 points over 100 consecutive non-training runs
after 4000 episodes of training.

To achieve this I used the following parameters:

    initial epsilon:    0.95
    epsilon decay:      0.998
    replay memory size: 5000
    train batch size:   512
    gamma:              0.99

"""

import gym
import sys
import random
import numpy as np
import tensorflow as tf
from network import DQN
from memory import Memory


class Actor:

    def __init__(self, session):
        self.env = gym.make("LunarLander-v2")
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.network = DQN(session, self.state_size, self.action_size)
        self.memory = Memory(5000)
        self.gamma = 0.99

    def episode(self, epsilon=1, train=True, render=False):

        state = self.env.reset()
        total_reward = 0

        while True:
            if render:
                self.env.render()
            
            if random.uniform(0.0, 1.0) < epsilon:
                action = random.randrange(0, self.action_size)
            else:
                action = self.network.predict_action(state)

            new_state, reward, done, _ = self.env.step(action)
            self.memory.add((state, action, reward, new_state))

            state = new_state
            total_reward += reward

            if done:
                break

        if train:
            self.train_dqn()

        return total_reward

    def train_dqn(self):

        batch = self.memory.sample(512)
        rewards = []

        # Calculate the expected reward for each transition
        for i in range(len(batch)):
            state, action, reward, new_state = batch[i]
            future_action = self.network.predict_action(new_state)
            future_reward = self.network.predict_value(new_state, future_action)
            expected_reward = reward + self.gamma * future_reward
            rewards.append(expected_reward)

        # Train the deep q network
        for i in range(len(rewards)):
            state, action, _, _ = batch[i]
            one_hot = [0 for _ in range(self.action_size)]
            one_hot[action] = 1
            self.network.train_dqn([state], [one_hot], [[rewards[i]]])


if __name__ == "__main__":

    with tf.Session() as sess:

        actor = Actor(sess)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # Test of pre-trained model
        if len(sys.argv) > 1 and sys.argv[1].lower() in ["solved", "s"]:
            saver.restore(sess, "./model/solved")
            total = 0
            for _ in range(10):
                score = actor.episode(-1, render=True, train=False)
            exit(0)

        # Parameters and tracking variables
        current_epsilon = .95
        epsilon_decay = 0.998
        total_score = 0
        last_10_scores = []

        # Train for 5000 episodes
        for ep in range(5000):

            score = actor.episode(current_epsilon, render=False)
            current_epsilon *= epsilon_decay

            total_score += score
            last_10_scores.append(score)
            if len(last_10_scores) > 10:
                last_10_scores.pop(0)

            print("Episode: {:5d} \tScore: {:10d} \tAvg: {:10d} \tAvg last 10 ep: {:10d}"
                  .format(ep, int(score), int(total_score/float(ep+1)), int(np.mean(last_10_scores))))

            if ep % 25 == 0 and ep > 0:
                path = saver.save(sess, "./tmp/model_"+str(ep)+".ckpt")
                print("\nCHECKPOINT", path, "\n")
