import numpy as np
import tqdm
from gridworld import GridWorld
from agents import *

def runGames(gameObj, learner, iters=100, game_speed=10, diagnostics=False):

	if diagnostics:
		# Fill in diagnostics
		pass

	# allow for game speed to change for inspection
	current_speed = game_speed

	for game_num in tqdm.trange(iters): 
		# make new game object
		game = gameObj(
			epoch=game_num,
			speed=current_speed,
			action_fn=learner.action_fn,
			reward_fn=learner.reward_fn
			)
		while game.game_loop():
			pass
		current_speed = game.rest

		if diagnostics:
			# Fill in diagnostics
			pass

		# reset learner
		learner.reset()
	return

if __name__ == '__main__':
	
	agent = RandomAgent()
	game = GridWorld

	runGames(GridWorld, agent, iters=20)

