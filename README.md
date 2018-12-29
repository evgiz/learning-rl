# Learning-RL

## Essential methods in reinforcement learning

> Sigve Rokenes, December 2018

Repo for keeping scripts written while researching the field. Public in case others interested in reinforcement learning are looking for code solutions. All scripts written in Python 3.6. Dependencies will probably be numpy and/or tensorflow for most scripts. Check the specific implementations for imports. All code released under the MIT license, so go wild! 

### Tabular Methods

#### Stationary k-armed bandit
Q learning in a stationary state space of probabilities length *k*, with a discrete action space *k*. Uses average reward over time for action value estimation. See `./bandit/k_bandit_stationary.py`

### Monte Carlo Methods

#### Blackjack

The actor learns to play blackjack near optimally after ~ 5000 games of experience. Solved using state-action estimation and policy improvement. Pseudo-code for the core learning algorithm is below. See `./monte_carlo/blackjack/blackjack_actor.py` for the full implementation.

```python
# State-action estimation
for s, a, r in experience:
	G = r + total_reward_after_s
	returns[(s,a)].append(G)
	q(s,a) = mean(returns[(s,a)])
	
# Policy estimation...
for s, _, _ in experience:
	policy(s) = argmax(q(s,a))
```

For this task I have also implemented a blackjack simulation environment which is used to train the actor. See `./monte_carlo/blackjack/blackjack_environment.py`
