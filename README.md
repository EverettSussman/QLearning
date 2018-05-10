# QLearning

## Introduction 

The purpose of this project is to better visualize how Q-learning algorithms behave under different hyperparameters.  To do this, we implemented GridWorld:

INSERT PICTURE OF GRIDWORLD HERE

GridWorld is a $2 \times 2$ board that shows the q-values for every possible state-action pair on the board.  The action that any agent takes for a given state is highlighted with orange text (note that if the agent is using an $\epsilon$-greedy policy, then it may take a random action).  

The rules of GridWorld are simple - an agent may move up, down, left, or right at any state.  If the agent moves out of bounds, then the agent loses 2 points.  If the agent moves into a red square, then the agent loses 10 points.  If the agent moves into the green square (the goal tile), then the agent wins 10 points.  

We have implemented the following agents to learn how to play GridWorld:

* RandomAgent - this agent moves randomly around the board, and is used as a baseline for our other agents.

* QLearningAgent - this agent learns how to play each board with an off-policy q-learning algorithm.   

* ApproximateQLearningAgent - this agent uses game-specific heuristics that utilize the state space to predict q-values.  (Currently, it is not well supported for GridWorld)

* DeepQAgent - this agent uses a deep q-learning neural network to learn how to play GridWorld.  Hyperparameters can still be tuned to optimize performance.  

## Repository Layout

## 


