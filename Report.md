# Implementation description

</br>

## Learning Algorithm

Multi DDPG agent algorithm with shared/separate replay buffer is used in this project. 

The agents at some point start to return nans for actions, so it does not learn.

Hyperparameters tested:
* Hidden units per layer from 64 to 1024
* Hidden layers from 1 to 4
* Replay batch size from 1 to 1024
* Update frequency from 1 to 1024
* TAU from  0.001 to 0.7
* Learning rate from 0.5 to 0.00005

Optimizers tested:
* Adam
* SGD

Configurations tested:
* Replay buffer with prioritization
* Noise

None of these combinations affected the agent's learning. Nans at some point start to appear anyway.

</br>

## Plot of Rewards
Plot of rewards can be seen after the environment has been solved.

No plot of rewards since agent is not learning. When nans appear avg score decreases to zero.

</br>

## Ideas for Future Work
Here's a list of optimizations that can be applied to the project:
1. Build an agent that finds the best hyperparameters.
