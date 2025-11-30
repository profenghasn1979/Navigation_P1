from abc import ABC, abstractmethod
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import make_q_network
from .buffer import ReplayBuffer
from .config import Config

device = torch.device("cpu")

class Agent(ABC):
    """Abstract Base Class for Agents."""
    
    @abstractmethod
    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy."""
        pass
    
    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        pass
        
    @abstractmethod
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples."""
        pass

class DQNAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, config: Config):
        """Initialize an Agent object.
        
        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            config (Config): configuration object
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.config = config

        # Q-Network
        self.qnetwork_local = make_q_network(state_size, action_size, seed, config.HIDDEN_LAYERS).to(device)
        self.qnetwork_target = make_q_network(state_size, action_size, seed, config.HIDDEN_LAYERS).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.config.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.config.GAMMA)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.
        
        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        if self.config.USE_DOUBLE_DQN:
            # Double DQN: Select action using local model, evaluate using target model
            indices = torch.argmax(self.qnetwork_local(next_states), dim=1, keepdim=True)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, indices)
        else:
            # Standard DQN
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config.TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
