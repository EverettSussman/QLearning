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

	def getVal(self, state, action):
		# return state,action values
		return self.qvalues[(state, action)]

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


class QLearningAgent(Agent):

	def __init__(self, actions=None, alpha=.1, alpha_dec=.99, 
									epsilon=.1, epsilon_dec=.95,
									gamma=.9, tol=.0001):
		Agent.__init__(self, actions=actions)
		"""
		Constructor for QLearning Agent

		Keyword Arguments

		alpha: learning rate (usually .1)
		gamma: discount factor
		tol: check for convergence of qvalues
		"""
		self.alpha = alpha
		self.alpha_dec = alpha_dec
		self.epsilon = epsilon
		self.epsilon_dec = epsilon_dec
		self.gamma = gamma
		self.tol = tol
		self.last_qvalues = None

	def reset(self):
		# Reset agent for next epoch
		self.last_state = None
		self.last_action = None
		self.last_reward = None

		self.alpha *= self.alpha_dec
		self.epsilon *= self.epsilon_dec

	def getAction(self, state):
		bestAction = None
		bestVal = float("-inf")

		for action in self.actions:
			tempVal = self.getVal(state, action)
			if tempVal > bestVal:
				bestVal = tempVal
				bestAction = action

		# Implement epsilon probability for choosing random action
		if np.random.random() < self.epsilon:
			bestAction = np.random.choice(self.actions)

		return bestAction, bestVal

	def learn(self, val):
		if self.last_state is None:
			return 

		last_val = self.getVal(self.last_state, self.last_action)
		
		old_stateAct = (self.last_state, self.last_action)
		r = self.last_reward
		correction = r - last_val + self.gamma * val

		self.qvalues[old_stateAct] += self.alpha * correction

	def action_fn(self, state):
		new_state = state
		new_action, bestVal = self.getAction(new_state)

		self.last_qvalues = self.qvalues

		self.learn(bestVal)

		self.last_action = new_action
		self.last_state = new_state

		return self.last_action
		

class ApproximateQLearningAgent(Agent):

	def __init__(self, actions=None, features=None, 
									alpha=.1, alpha_dec=.99,
									epsilon=.1, epsilon_dec=.95, 
									gamma=.9, tol=.0001):
		Agent.__init__(self, actions=actions)
		"""
		Constructor for ApproximateQLearning Agent

		Keyword Arguments

		features: list of functions to use for approximate 
			q-learning agent
		alpha: learning rate (usually .1)
		gamma: discount factor
		tol: check for convergence of qvalues
		"""
		self.alpha = alpha
		self.alpha_dec = alpha_dec
		self.epsilon = epsilon
		self.epsilon_dec = epsilon_dec
		self.gamma = gamma
		self.tol = tol
		self.last_qvalues = None

		if features is None:
			raise NameError("No features provided to ApproximateQLearningAgent")
		self.features = features + [1]
		self.fvals = None
		self.weights = np.zeros(len(self.features))

	def reset(self):
		# Reset agent for next epoch
		self.last_state = None
		self.last_action = None
		self.last_reward = None

		self.alpha *= self.alpha_dec
		self.epsilon *= self.epsilon_dec

	def getVal(self, state, action):
		# add back constant value for bias
		self.fvals = np.array([feature(state, action) for 
								feature in self.features if feature != 1] + [1])
		return np.dot(self.weights, self.fvals)

	def getAction(self, state):
		bestAction = None
		bestVal = float("-inf")

		for action in self.actions:
			tempVal = self.getVal(state, action)
			if tempVal > bestVal:
				bestVal = tempVal
				bestAction = action

		if np.random.random() < self.epsilon:
			bestAction = np.random.choice(self.actions)

		return bestAction, bestVal

	def learn(self, val):
		if self.last_state is None:
			return 

		last_val = self.getVal(self.last_state, self.last_action)
		r = self.last_reward
		correction = r + self.gamma * val - last_val
		self.weights += self.alpha * correction * self.fvals
		self.weights = self.weights / np.linalg.norm(self.weights)

	def action_fn(self, state):
		new_state = state
		new_action, bestVal = self.getAction(new_state)
		print(self.weights)
		self.last_qvalues = self.qvalues
		self.learn(bestVal)

		self.last_action = new_action
		self.last_state = new_state
		return self.last_action














