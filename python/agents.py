import numpy as np
from collections import Counter

class Agent():

	def __init__(self, actions=None):

		"""
		Constructor for Random Agent Learner

		Keyword Arguments:

			actions: If None - uses gridworld actions as default.
		Otherwise, must be a list of valid actions for the game.
		"""

		# Initialize state action values
		self.last_state = None
		self.last_action = None
		self.last_reward = None
		self.qvalues = Counter()

		# Initialize action list
		if actions is None:
			print("Using gridworld actions")
			self.actions = ["RIGHT", "LEFT", "UP", "DOWN"]
		else:
			self.actions = actions


	def reset(self):
		# Reset agent for next epoch
		self.last_state = None
		self.last_action = None
		self.last_reward = None

	def action_fn(self, state):
		# Handle actions
		pass

	def reward_fn(self, reward):
		# Handle rewards
		self.last_reward = reward


class RandomAgent(Agent):

	def action_fn(self, state):
		# Return a random action
		new_action = np.random.choice(self.actions)
		new_state = state

		self.last_action = new_action
		self.last_state = new_state
		return self.last_action










