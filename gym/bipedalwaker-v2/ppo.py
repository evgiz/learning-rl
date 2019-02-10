
"""
Author: Sigve Rokenes
Date: February, 2019

Proximal Policy Optimization with objective clipping

"""

import numpy as np
import tensorflow as tf


# ================================= #
#                                   #
#    Proximal Policy Optimization   #
#                                   #
# ================================= #

class PPO:

    def __init__(self, state_size, action_size, clip_low=-1, clip_high=1, gamma=0.95, lam=0.8):

        self.state_size = state_size
        self.action_size = action_size

        # ========================= #
        #     Hyper parameters      #
        # ========================= #

        self.gamma = gamma
        self.lam = lam
        self.entropy_factor = 0.0001
        learning_rate = 0.0001
        v_loss_coeff = 0.5
        epsilon = 0.2

        # ========================= #
        #          Model            #
        # ========================= #

        self.state_input = tf.placeholder(tf.float32, shape=[None, state_size], name="state_input")
        self.target_action = tf.placeholder(tf.float32, shape=[None, action_size], name="target_action")
        self.log_prob_old = tf.placeholder(tf.float32, shape=[None, action_size], name="log_prob_old")
        self.target_reward = tf.placeholder(tf.float32, shape=[None], name="target_reward")
        self.advantages = tf.placeholder(tf.float32, shape=[None], name="target_advantage")

        self.mu, self.sigma, self.critic = self.build_network()
        normal_dist = tf.distributions.Normal(self.mu, tf.exp(self.sigma))
        sample = normal_dist.sample(1)
        self.actor = tf.clip_by_value(sample, clip_low, clip_high)

        # ========================= #
        #        Optimization       #
        # ========================= #

        self.log_prob = normal_dist.log_prob(self.target_action)
        policy_ratio = tf.reduce_mean(tf.exp(self.log_prob - self.log_prob_old), axis=1, keep_dims=True)
        change_adv = policy_ratio * tf.reshape(self.advantages, [-1, 1])
        change_eps = tf.clip_by_value(policy_ratio, 1.0 - epsilon, 1.0 + epsilon) * tf.reshape(self.advantages, [-1, 1])
        clip_delta_param = tf.minimum(change_adv, change_eps)
        self.policy_loss = -tf.reduce_mean(clip_delta_param)

        self.entropy = tf.reduce_sum(normal_dist.entropy(), axis=-1) * self.entropy_factor
        self.value_loss = v_loss_coeff * tf.losses.mean_squared_error(tf.squeeze(self.critic), self.target_reward)

        loss = self.value_loss + self.policy_loss - self.entropy

        weights = [v for v in tf.trainable_variables()]
        grads = tf.gradients(loss, weights)
        grads = zip(grads, weights)
        adam = tf.train.AdamOptimizer(learning_rate)
        self.optimize = adam.apply_gradients(grads)

    # ========================= #
    #         Network           #
    # ========================= #

    def build_network(self):

        net = tf.layers.dense(self.state_input, 512, activation=tf.nn.relu)
        net = tf.layers.dense(net, 512, activation=tf.nn.relu)

        with tf.variable_scope("actor"):
            mu = tf.layers.dense(net, self.action_size,  activation=tf.nn.tanh)
            sigma = tf.layers.dense(net, self.action_size, activation=tf.nn.softplus)

        with tf.variable_scope("critic"):
            critic = tf.layers.dense(net, 1)

        return mu, sigma, critic

    # ========================= #
    #        Prediction         #
    # ========================= #

    def act(self, sess, state):
        act = self.actor_predict(sess, state)
        return act

    def actor_predict(self, sess, state):
        return sess.run(self.actor, feed_dict={
            self.state_input: state
        })[0][0]

    def critic_predict(self, sess, state):
        return sess.run(self.critic, feed_dict={
            self.state_input: state
        })[0][0]

    def calc_values(self, sess, states):
        return sess.run(self.critic, feed_dict={
            self.state_input: states
        })

    # ========================= #
    #         Training          #
    # ========================= #

    def calc_returns(self, rewards, dones):
        r = 0
        returns = []
        for i in reversed(range(len(rewards))):
            r = rewards[i] + (1.0 - dones[i]) * self.gamma * r
            returns.append(r)
        returns.reverse()
        return returns

    def calc_advantages(self, returns, values):
        advantages = [r - v for r, v in zip(returns, values)]
        for i in reversed(range(len(advantages) - 1)):
            advantages[i] = advantages[i] + self.gamma * self.lam * advantages[i + 1]
        return advantages

    def train(self, sess, states, actions, rewards, new_states, dones):
        assert len(states) == len(actions) == len(rewards) == len(new_states), "Invalid training data"

        returns = self.calc_returns(rewards, dones)
        values = self.calc_values(sess, states).reshape([1, -1])[0]
        advantages = self.calc_advantages(returns, values)
        advantages = np.divide(np.subtract(advantages, np.mean(advantages)), np.std(advantages))

        log_probs = sess.run(self.log_prob, feed_dict={self.state_input: states, self.target_action: actions})

        c_loss, p_loss, _ = sess.run([self.value_loss, self.policy_loss, self.optimize], feed_dict={
            self.state_input: states,
            self.target_action: actions,
            self.target_reward: returns,
            self.advantages: advantages,
            self.log_prob_old: log_probs
        })

        return p_loss, c_loss
