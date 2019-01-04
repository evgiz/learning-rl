
import tensorflow as tf
import numpy as np


class DQN:

    def __init__(self, sess, state_size, action_size):

        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size

        self.p_state = tf.placeholder(tf.float32, shape=[None, state_size])
        self.p_action = tf.placeholder(tf.float32, shape=[None, action_size])
        self.p_reward = tf.placeholder(tf.float32, shape=[None, 1])

        state_dense = tf.layers.dense(
            self.p_state, 256, activation=tf.nn.relu
        )
        action_dense = tf.layers.dense(
            self.p_action, 256, activation=tf.nn.relu
        )

        combined = tf.multiply(state_dense, action_dense)

        hl1 = tf.layers.dense(combined, 512, activation=tf.nn.relu)
        dp = tf.layers.dropout(hl1, 0.25)
        hl2 = tf.layers.dense(dp, 512, activation=tf.nn.relu)
        dp2 = tf.layers.dropout(hl2, 0.25)
        self.model = tf.layers.dense(dp2,1)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.model, self.p_reward))
        self.train = self.optimizer.minimize(self.loss)

    def predict_action(self, state):
        rewards = [0 for _ in range(self.action_size)]
        for action in range(self.action_size):
            rewards[action] = self.predict_value(state, action)
        return np.argmax(rewards)

    def predict_value(self, state, action):
        one_hot = [0 for _ in range(self.action_size)]
        one_hot[action] = 1
        rewards = self.sess.run(self.model, feed_dict={
            self.p_state: np.array([state]),
            self.p_action: np.array([one_hot])
        })
        return rewards[0][0]

    def train_dqn(self, state, action, reward):
        _, loss_value = self.sess.run([self.train, self.loss], feed_dict={
            self.p_state: np.array(state),
            self.p_action: np.array(action),
            self.p_reward: np.array(reward)
        })
        return loss_value
