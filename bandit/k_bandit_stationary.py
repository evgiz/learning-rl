"""
k-Bandit Problem

A very simple reinforcement learning problem.
The state is stationary, and consists of k probabilities of a reward.
The actor attempts to maximize its reward, while exploring the action space.
Exploration rate is defined by epsilon.

Sigve Røkenes
December, 2018
"""

import random
import numpy as np
import statistics


def k_bandit(k=10, epsilon=0.3, steps=200):

    print("Running {}-bandit with {} steps and epsilon {}...".format(k, steps, epsilon))

    Q = [0 for _ in range(k)]
    N = [0 for _ in range(k)]

    bandit = [random.uniform(0, 1) for _ in range(k)]
    best_choice = np.argmax(bandit, 0)

    total_best_choice = 0
    total_reward = 0

    for _ in range(steps):
        if random.uniform(0, 1) < epsilon:
            choice = random.randrange(0, k)
        else:
            choice = np.argmax(Q, 0)
        reward = 1 if random.uniform(0, 1) < bandit[choice] else 0
        total_reward += reward
        if choice == best_choice:
            total_best_choice += 1
        N[choice] += 1
        Q[choice] += 1.0/float(N[choice]) * (reward - Q[choice])

    mean_error = statistics.mean(abs(Q[i]-bandit[i]) for i in range(k))
    mean_reward = statistics.mean(bandit) * steps
    print("\nModel error:\t{:.2f}%".format(mean_error*100))
    print("Mean reward:\t{:.2f}".format(mean_reward))
    print("\nBest choice:\t{:.2f}%".format(total_best_choice/float(steps)*100))
    print("Reward earned:\t{:.2f}".format(total_reward))
    print("\nBandit:\t", ["{:.4f}".format(val) for val in bandit])
    print("Q:\t\t", ["{:.4f}".format(val) for val in Q])
    print("N:\t\t", ["{:06d}".format(val) for val in N])


if __name__ == "__main__":
    k_bandit()