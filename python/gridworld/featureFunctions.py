import numpy as np

GOALVAL = 10
"""
State information to use.
Format is agentLoc (x,y), goal location (x,y)
"""

def feature1(state, action):
	""" 
	This feature represents the distance 
	the agent is to the goal.  It returns
	1/dist, which is small for large distances
	and large for small distances.
	"""
	agentpos = np.array([state[0], state[1]])
	goalpos = np.array([state[2], state[3]])
	dist = np.linalg.norm(agentpos - goalpos)

	if action == "UP" and state[3] < state[1]:
		dist -= .1
	elif action == "RIGHT" and state[2] > state[0]:
		dist -= .1
	elif action == "DOWN" and state[3] > state[1]:
		dist -= .1
	elif action == "LEFT" and state[2] < state[0]:
		dist -= .1
	else:
		dist += .1

	return -.1 * dist

def feature2(state, action):
	"""
	This feature penalizes moving towards obstacles
	or bad red squares.
	"""
	agentpos = np.array([state[0], state[1]])

	if action == "UP":
		agentpos[1] -= 1
	if action == "RIGHT":
		agentpos[0] += 1
	if action == "DOWN":
		agentpos[1] += 1
	if action == "LEFT":
		agentpos[0] -= 1

	statePad = 6
	boardRows = state[4]
	boardCols = state[5]
	stateIndex = statePad + agentpos[1] * boardCols + agentpos[0]

	if agentpos[0] >= boardCols or agentpos[0] < 0:
		return -.5
	elif agentpos[1] >= boardRows or agentpos[1] < 0:
		return -.5
	elif state[stateIndex] is None:
		return -.5
	elif state[stateIndex] == -10:
		return -1.0
	else:
		return -.1

def feature3(state, action): 
	"""
	Get out of nook for map 0
	"""
	agentpos = np.array([state[0], state[1]])
	if (agentpos[0] == 0 and (agentpos[1] == 0 or agentpos[1] == 1) 
		and action == "DOWN"):
		return 10
	else:
		return 0

gridWorldFeatures = [feature1]


