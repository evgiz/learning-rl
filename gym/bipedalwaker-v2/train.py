
"""
Author: Sigve Rokenes
Date: February, 2019

Trainer for PPO

"""

import gym
import numpy as np
import tensorflow as tf
from ppo import PPO


class Trainer:

    def __init__(self, environment):
        self.env = gym.make(environment)
        self.max_steps = 2000
        a_size = self.env.action_space.shape[0]
        s_size = self.env.observation_space.shape[0]
        a_low = self.env.action_space.low
        a_high = self.env.action_space.high
        self.ppo = PPO(s_size, a_size, a_low, a_high)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.session()

    def session(self):
        for it in range(3000):
            print("Epoch", it)
            scores = self.run(50)
            print(np.mean(scores))
            self.save(it)

    def save(self, name, path="model/"):
        self.saver.save(self.sess, path+"model_" + str(name))

    def run(self, episodes=1, render=False, train=True):
        rewards = []
        for ep in range(episodes):
            ts, ta, tr, tns, dns = self.t_gen(render)
            rewards.append(sum(tr))
            if train:
                _, _ = self.ppo.train(self.sess, ts, ta, tr, tns, dns)
        return rewards

    def t_gen(self, render=False):
        ts, ta, tr, tns, dns = [], [], [], [], []
        cs = self.env.reset()
        for _ in range(self.max_steps):
            if render:
                self.env.render()
            act = self.ppo.act(self.sess, cs.reshape([1, -1]))
            ns, rw, dn, _ = self.env.step(act)
            ts.append(cs)
            ta.append(act)
            tr.append(rw)
            tns.append(ns)
            dns.append(float(dn))
            cs = ns
            if dn:
                break
        return ts, ta, tr, tns, dns


if __name__ == "__main__":
    Trainer("BipedalWalker-v2")


