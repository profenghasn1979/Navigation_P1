import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=(64, 64)):
        """Initialize parameters and build model.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (tuple): Number of nodes in each hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Create layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(state_size, hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            
        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], action_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)

def make_q_network(state_size, action_size, seed, hidden_layers=(64, 64)):
    """Factory function to create a QNetwork.
    
    Args:
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        hidden_layers (tuple): Number of nodes in each hidden layer
        
    Returns:
        QNetwork: The initialized Q-Network
    """
    return QNetwork(state_size, action_size, seed, hidden_layers)
