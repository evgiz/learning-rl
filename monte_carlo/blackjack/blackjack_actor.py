"""
=================================

Author: Sigve Rokenes
Date: December, 2018

The following code attempts to train a blackjack player
to win using the Monte-Carlo method.

=================================
           ALGORITHM
=================================

The algorithm works as follows:

    1. Run an episode using the current policy and record transitions in memory - MonteCarloLearner.__run()
    2. Improve the state-action estimation using experiences:
        - for every (s, a) in memory:
            - calculate expected reward
            - add reward to returns[(s,a)]
            - update state-action function q[s][a] = mean(returns[(s,a)])
    3. Improve the policy based on the state-action estimation:
        - for every s in memory:
            - policy[s] = argmax(q[s])

=================================

Function explanations:

    train() - trains the policy using the above algorithm
        parameters:
            episodes:       the number of games to play
            verbose:        print verbose details
            epsilon:        exploration vs. exploitation rate
            epsilon_decay:  decay of epsilon for each step

    evaluate_trained_actor() - tests the actor using greedy choices (eg. the best known policy)

    evaluate_random_actor() - tests an actor which choses randomly every time

    evaluate_basic_actor() - tests an actor with a hard coded simple policy
        If the cards have a sum of <= 16, hit, otherwise stick
        (assumes ace is used as the best choice, eg. 11 if the sum is < 21)


Finally, the play_game() function can be used to test the environment with user inputs.

=================================
            RESULTS
=================================

The code has proved learning capability, and achieves a win ratio of ~ 42% after training for 5000 games.
This is on par with the best possible policy for the game. Note that the percentage will vary
every training session, and the stated percentage was from a 10000 episode evaluation.

The results were achieved using the following parameters:

    epsilon = 1
    epsilon_decay = 0.00015
    episodes = 5000

"""

from blackjack_environment import BlackjackEnvironment
import numpy as np
import random

class MonteCarloLearner:

    def __init__(self):
        self.env = BlackjackEnvironment()
        self.policy_dict = {}
        self.qa_dict = {}
        self.returns = {}

    def train(self, episodes=1000, verbose=False, epsilon=1, epsilon_decay=1e-4):
        self.__run(episodes, verbose, method="explore",  epsilon=epsilon, epsilon_decay=epsilon_decay)

    def evaluate_trained_actor(self, episodes=1000, verbose=False):
        self.__run(episodes, verbose, method="greedy")

    def evaluate_random_actor(self, episodes=1000, verbose=False):
        self.__run(episodes, verbose, method="random")

    def evaluate_basic_actor(self, episodes=1000, verbose=False):
        self.__run(episodes, verbose, method="basic")

    def __run(self, episodes=100, verbose=False, method="explore", epsilon=1, epsilon_decay=0):

        results = [0, 0, 0]

        for episode in range(episodes):

            memory = []
            state, reward, done = self.env.reset()

            # Blackjack!
            if done and reward == 1:
                results[2] += 1
                continue

            if verbose:
                print("\n=== Game {} ===".format(episode))
                print(self.env.summary())

            while True:

                # Chose an action based on method
                if method == "basic":
                    action = self.env.basic_strategy()
                else:
                    if method == "random" or state not in self.policy_dict:
                        action = random.randrange(0, 2)
                    elif method == "explore" and random.uniform(0, 1) < epsilon:
                        action = random.randrange(0, 2)
                    else:
                        action = self.policy_dict[state]

                epsilon *= (1-epsilon_decay)

                if verbose:
                    print("Hit" if action == 1 else "Stick")

                new_state, reward, done = self.env.step(action)
                memory.append((state, action, reward))
                state = new_state

                if verbose:
                    print(self.env.summary())

                if done:
                    result_out = ["Win", "Draw", "Loss"]
                    results[reward+1] += 1
                    if verbose:
                        print(result_out[reward+1])
                    break

            if method == "explore":
                self.__train(memory)

        print(method.upper()+" Monte-Carlo complete!\n")
        print("\tEpisodes:\t{}".format(episodes))
        print("\tLoss:    \t{:5.2f}%".format(results[0] / float(episodes) * 100))
        print("\tWin:     \t{:5.2f}%".format(results[2] / float(episodes) * 100))
        print("\tDraw:    \t{:5.2f}%\n".format(results[1] / float(episodes) * 100))

    def __train(self, memory):

        # State-action improvement
        for i in range(len(memory)):
            state, action, reward = memory[i]

            # Calculate expected reward
            expected_reward = reward
            for j in range(i+1, len(memory)):
                _, _, next_reward = memory[j]
                expected_reward += next_reward

            # Add to returns
            if (state, action) not in self.returns:
                self.returns[(state, action)] = []
            self.returns[(state, action)].append(expected_reward)

            # Update state-action prediction
            if state not in self.qa_dict:
                self.qa_dict[state] = [0, 0]
            mean_reward = np.mean(self.returns[(state, action)])
            self.qa_dict[state][action] = mean_reward

        # Policy improvement
        for i in range(len(memory)):
            state, _, _ = memory[i]
            best_qa = np.argmax(np.array(self.qa_dict[state]))
            self.policy_dict[state] = best_qa


def play_game(games=1):

    import time

    env = BlackjackEnvironment()

    for _ in range(games):

        _, reward, _ = env.reset()
        print(env.summary())

        if reward == 1:
            print("BLACKJACK!\n")
            continue

        while True:

            action = None
            while action != "h" and action != "s":
                action = input("Hit/Stick [h/s]: ")
            action = 1 if action == "h" else 0

            _, reward, done = env.step(action)
            print(env.summary())

            if done:
                result = ["You Lose!", "Draw", "You Win!"]
                print(result[reward+1], "\n")
                time.sleep(.25)
                break


if __name__ == "__main__":

    print("\nRunning Monte-Carlo blackjack...\n")
    learner = MonteCarloLearner()
    learner.train(5000, epsilon_decay=0.00015)
    learner.evaluate_trained_actor(10000)
