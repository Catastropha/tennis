# Implementation description

</br>

## Learning Algorithm

Multi DDPG agent algorithm with shared/separate replay buffer is used in this project. 

Configurations:
* 2 hidden layers with 512 and 256 hidden units for both actor and critic
* Shared replay buffer
* Replay batch size 512
* Buffer size 1e6
* Replay without prioritization
* Update frequency 4
* TAU from  1e-3
* Learning rate 1e-4 for actor and 3e-4 for critic
* Ornstein-Uhlenbeck noise
* 20% droput for critic


</br>

## Plot of Rewards
Plot of rewards can be seen after the environment has been solved.

Environment solved in 59 episodes.

</br>

## Ideas for Future Work
Here's a list of optimizations that can be applied to the project:
1. Build an agent that finds the best hyperparameters for an agent
2. Prioritization for replay buffer
3. Paramter space noise for better exploration
4. Test shared network between agents
