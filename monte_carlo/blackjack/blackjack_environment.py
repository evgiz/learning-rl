
"""
=================================
Author: Sigve Rokenes
Date: December, 2018
=================================

=================================
    BLACKJACK ENVIRONMENT
=================================

The following code attempts to simulate a blackjack
game environment for use in reinforcement learning.

The action space has two discrete actions:

    0: Stick (stick with your cards, wait for dealer)
    1: Hit (receive an additional card)

The state space is discrete and consists of a tuple:

    (has_usable_ace, player_sum, dealer_sum)

I recommend reading up on the rules of blackjack if you are unfamiliar.
Note the environment is untested and may have inconsistencies.

=================================
            USAGE
=================================

Before each episode, call env.reset()
This resets the state to two player cards and one dealer card.

For each step, call env.step(action), where action is either 0 or 1.
The function returns a tuple state, reward, and a boolean done

Every step returns a reward of zero, except when the game is done. Then:
    -1:  indicates a loss
     0:  indicates a draw
     1:  indicates a win

env.basic_strategy() can be used to get an action following a basic strategy for the current state:
    If the cards have a sum of <= 16, hit, otherwise stick
    (assumes ace is used as the best choice, eg. 11 if the sum is < 21)

    This strategy could be used for more guided learning, or as a baseline for testing.

env.summary() returns a string representation of the current state.

"""

import random


class BlackjackEnvironment:

    __player_cards = []
    __dealer_cards = []

    __card_types = 13

    def __init__(self):
        self.reset()

    def reset(self):
        self.__player_cards = [
            self.__pick_card(),
            self.__pick_card()
        ]
        self.__dealer_cards = [
            self.__pick_card()
        ]

        state = self.__calculate_discrete_state()

        # Blackjack!
        if self.__is_blackjack(self.__player_cards):
            # Check if dealer gets blackjack...
            dealer_blackjack = False
            while not self.__is_bust(self.__dealer_cards):
                self.__dealer_cards.append(self.__pick_card())
                if self.__is_blackjack(self.__dealer_cards):
                    dealer_blackjack = True
                    break
            return state, 1 if not dealer_blackjack else 0, True

        return state, 0, False

    def step(self, hit):

        if hit > 0:
            self.__player_cards.append(self.__pick_card())

            reward = 0
            done = False

            if self.__is_bust(self.__player_cards):
                reward = -1
                done = True

            if self.__dealer_strategy(self.__dealer_cards) == 0:
                reward = self.__calculate_reward()
                done = True

            state = self.__calculate_discrete_state()
            return state, reward, done
        else:
            # Dealer method
            while self.__dealer_strategy(self.__dealer_cards) == 1:
                self.__dealer_cards.append(self.__pick_card())

            if self.__is_bust(self.__dealer_cards):
                reward = 1
            else:
                reward = self.__calculate_reward()

            state = self.__calculate_discrete_state()
            return state, reward, True

    def basic_strategy(self):
        return self.__dealer_strategy(self.__player_cards)

    def __dealer_strategy(self, cards):
        da, db = self.__calculate_sum(cards)

        if db <= 16 or (da <= 16 and db > 21):
            return 1
        else:
            return 0

    def summary(self):

        player = ""
        for card in self.__player_cards:
            player += "({})".format(card)
        dealer = ""
        for card in self.__dealer_cards:
            dealer += "({})".format(card)

        da, db = self.__calculate_sum(self.__dealer_cards)
        pa, pb = self.__calculate_sum(self.__player_cards)

        summary = "Dealer: "+dealer+" = "+str(db if db <= 21 else da)
        summary += "\nPlayer: "+player+" = "+str(pb if pb <= 21 else pa)

        return summary

    def __calculate_reward(self):

        pa, pb = self.__calculate_sum(self.__player_cards)
        da, db = self.__calculate_sum(self.__dealer_cards)

        player_diff = min(abs(pa - 21), abs(pb - 21))
        dealer_diff = min(abs(da - 21), abs(db - 21))

        if player_diff == dealer_diff:
            reward = 0
        elif player_diff < dealer_diff:
            reward = 1
        else:
            reward = -1

        return reward

    def __calculate_discrete_state(self):
        player_min, player_max = self.__calculate_sum(self.__player_cards)
        dealer_sum = min(10, self.__dealer_cards[0])

        has_ace = self.__player_cards.count(1) > 0
        ace_usable = 1 if (has_ace and player_max <= 21) else 0
        player = player_max if ace_usable else player_min

        return ace_usable, player, dealer_sum

    def __is_blackjack(self, cards):
        a, b = self.__calculate_sum(cards)
        return a == 21 or b == 21

    def __is_bust(self, cards):
        a, _ = self.__calculate_sum(cards)
        return a > 21

    def __calculate_sum(self, cards):
        min_sum = 0
        max_sum = 0

        for card in cards:
            value = 10 if card >= 10 else card
            min_sum += value
            max_sum += 11 if card == 1 else value

        return min_sum, max_sum

    def __pick_card(self):
        return random.randrange(0, self.__card_types) + 1
