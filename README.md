# tennis
Udacity deep reinforcement learning multi-agent tennis project

</br>

## Project details
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

</br>

## Getting started
The project uses Jupyter Notebook.
This command needs to be run to install the needed packages:

```
!pip -q install ./python
```
Running all the cells in the notebook will install it automatically.

</br>

## Instructions
The project consists of 9 files:
* tennis.ipynb - run this file in Jupyter Notebook
* ma.py - the Multi Agent class
* agent.py - the DDPG Agent class
* network.py - the Actor and Critic models
* memory.py - the replay buffer class
* noise.py - the noise class
* config.py - the configuration class
* checkpoint.pth - saved trained model to use
* Report.md - description of the implementation

Use Config() class to define all project configurations (hyperparameters, network, optimizers etc.)
