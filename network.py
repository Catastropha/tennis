import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, state_size, hidden_units, seed, gate=F.relu, final_gate=F.tanh):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes for layers
            seed (int): Random seed
            gate (function): activation function
            final_gate (function): final activation function
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        self.final_gate = final_gate
        self.normalizer = nn.BatchNorm1d(state_size)
        dims = (state_size, ) + hidden_units
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.output = nn.Linear(dims[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.weight.data.uniform_(-1, 1)
        self.output.weight.data.uniform_(-1, 1)


    def forward(self, states):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.normalizer(states)
        for layer in self.layers:
            x = self.gate(layer(x))
        return self.final_gate(self.output(x))

    
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, action_size, state_size, hidden_units, seed, gate=F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes for layers
            seed (int): Random seed
            gate (function): activation function
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        self.normalizer = nn.BatchNorm1d(state_size)
        dims = (state_size, ) + hidden_units
        self.layers = nn.ModuleList()
        count = 0
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            if count == 1:
                self.layers.append(nn.Linear(dim_in+action_size, dim_out))
            else:
                self.layers.append(nn.Linear(dim_in, dim_out))
            count += 1
        self.output = nn.Linear(dims[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.weight.data.uniform_(-1, 1)
        self.output.weight.data.uniform_(-1, 1)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.normalizer(states)
        x = self.layers[0](x)
        x = torch.cat((x, actions), dim=1)
        for i in range(1, len(self.layers)):
            x = self.gate(self.layers[i](x))
        return self.output(x)
