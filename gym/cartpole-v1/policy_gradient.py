
import gym
import numpy as np
import tensorflow as tf
import sys


class PolicyGradient:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.state_input = tf.placeholder(tf.float32, shape=[None, state_size], name="state_input")
        self.action_input = tf.placeholder(tf.float32, shape=[None, action_size], name="action_input")
        self.reward_input = tf.placeholder(tf.float32, shape=[None, ], name="reward_input")

        hl1 = tf.layers.dense(self.state_input, 32, activation=tf.nn.relu)
        hl2 = tf.layers.dense(hl1, 16, activation=tf.nn.relu)
        hl3 = tf.layers.dense(hl2, 12, activation=tf.nn.relu)
        out = tf.layers.dense(hl3, action_size)

        self.policy = tf.nn.softmax(out)

        softmax_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=self.action_input)
        loss = tf.reduce_mean(softmax_entropy * self.reward_input)

        self.train = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

    def predict(self, sess, state, greedy=False):
        probabilities = sess.run(self.policy, feed_dict={
            self.state_input: state.reshape([1, self.state_size])
        })[0]
        if greedy:
            return np.argmax(probabilities)
        return np.random.choice(range(self.action_size), p=probabilities)

    def update_weights(self, sess, states, actions, rewards):
        sess.run(self.train, feed_dict={
            self.state_input: states,
            self.action_input: actions,
            self.reward_input: rewards
        })


class Worker:

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.policy_gradient = PolicyGradient(self.state_size, self.action_size)
        self.gamma = 0.97

    def episode(self, sess, train=True, greedy=False, render=False):

        states = []
        actions = []
        rewards = []

        state = self.env.reset()

        while True:
            if render:
                self.env.render()

            action = self.policy_gradient.predict(sess, state, greedy)
            new_state, reward, done, _ = self.env.step(action)

            states.append(state)

            one_hot = np.zeros(self.action_size)
            one_hot[action] = 1

            actions.append(one_hot)
            rewards.append(reward)

            state = new_state

            if done:
                if train:
                    expected_reward = 0.0
                    target_vector = np.zeros(len(rewards))
                    for i in reversed(range(len(rewards))):
                        expected_reward = expected_reward * self.gamma + rewards[i]
                        target_vector[i] = expected_reward

                    states = np.vstack(np.array(states))
                    actions = np.vstack(np.array(actions))
                    target_vector = (target_vector - np.mean(target_vector)) / (np.std(target_vector))
                    self.policy_gradient.update_weights(sess, states, actions, target_vector)

                break

        return np.sum(rewards)


if __name__ == "__main__":

    print("CartPole-v1 - Policy Gradient Ascent")
    worker = Worker()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if len(sys.argv) > 1:
            if sys.argv[1].lower() in ["solved", "s"]:
                saver.restore(sess, "model/solved")
                for _ in range(10):
                    worker.episode(sess, train=False, greedy=True, render=True)
                exit(0)

        scores = []
        for episode in range(500):
            scr = worker.episode(sess)
            scores.append(scr)

            if episode % 10 == 0:
                print("({}):\t{}".format(episode, np.mean(scores)))
                scores.clear()

            if episode % 50 == 0:
                path = saver.save(sess, "./tmp/model_" + str(episode) + ".ckpt")

        for _ in range(10):
            worker.episode(sess, train=False, greedy=True, render=True)