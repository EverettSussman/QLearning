import numpy as np
import pickle
import matplotlib.pyplot as plt
import tqdm
from gridworld import *
from agents import *

def runGames(gameObj, learner, bare=False, iters=100, game_speed=1000, game_params=None, diagnostics=False):

    if diagnostics:
        # Fill in diagnostics
        ret = {
        'scores': [],
        'alphas': [],
        'epsilons': []
        }

        if not bare:
            # live plot for diagnostic values
            plt.ion()
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            plt.subplots_adjust(hspace=0.7)
            plt.suptitle("Game Diagnostics")
            plt.show()

    # allow for game speed to change for inspection
    current_speed = game_speed

    for game_num in range(iters): 
        # make new game object
        game = gameObj(
            params=game_params,
            epoch=game_num,
            speed=current_speed,
            learner=learner
            )

        if bare:
            while game.bare_game_loop():
                pass
        else:
            while game.game_loop():
                pass

        current_speed = game.rest

        if diagnostics:
            # Fill in diagnostics
            ret['scores'].append(game.score)
            ret['alphas'].append(learner.alpha)
            if hasattr(learner, 'epsilon'):
                ret['epsilons'].append(learner.epsilon)

            if not bare:
                ax1.plot(ret['scores'], c="k")
                ax1.set_title("Score", loc="left")

                ax2.plot(ret['alphas'], c="k")
                ax2.set_title(r"$\alpha$", loc="left")

                ax3.plot(ret['epsilons'], c="k")
                ax3.set_title(r"$\epsilon$", loc="left")

                # ax4.plot(log["q_update_size"], c="k")
                # ax4.set_title("Q-update size", loc="left")

                # ax5.plot(log["eta"], c="k")
                # ax5.set_title(r"$\eta$", loc="left")

                # ax6.plot(log["average_rewards"], c="k")
                # ax6.set_title(r"$\bar{r}$", loc="left")

                plt.draw()
            else:
                pass
                # print("Epoch: {}".format(game_num))
                # print("Score: {}".format(game.score))


        # reset learner
        learner.reset()
    return ret, learner

def genScoreDist(filename, reps, epochs, game, agent_class, params, features=None):
    totScores = []

    for _ in tqdm.trange(reps):
        if features is not None:
            agent = agent_class(features=features)
        else:
            agent = agent_class()
        results, run_agent = runGames(game, agent, bare=True, iters=epochs, game_params=params, diagnostics=True)
        totScores.append(results['scores'])
        np.save(filename, totScores)
    return totScores

def showLearner(epochs, agent_class, game, params, bare=False, features=None):
    if features is not None:
        agent = agent_class(features=features)
    else:
        agent = agent_class()
    results, run_agent = runGames(game, agent, bare=bare, iters=epochs, game_params=params, diagnostics=True)
    return results, run_agent

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# # sample usage
# save_object(company1, 'company1.pkl')

if __name__ == '__main__':

    # print(gridWorldFeatures)
    
    # # Save runs 100 times to generate nice score graphs
    params = {'probs':[.1, .8, .1], 'board_file': 'gridworld/maps/map0.board'}
    # genScoreDist('gridworld/results/gridworld4.npy', 100, 100, GridWorld, QLearningAgent, params)
    # totScores = []
    features = gridWorldFeatures
    agent = DeepQAgent
    game = GridWorld
    genScoreDist('gridworld/results/DQN0.npy', 20, 1000, game, agent, params)
    # res, run_agent = showLearner(1000, agent, game, params, bare=True, features=None)
    













