from abc import ABC, abstractmethod
from unityagents import UnityEnvironment
import numpy as np

class BaseEnv(ABC):
    """Abstract Base Class for Environments."""
    
    @abstractmethod
    def reset(self):
        """Resets the environment and returns the initial state."""
        pass
    
    @abstractmethod
    def step(self, action):
        """Takes a step in the environment.
        
        Returns:
            next_state, reward, done, info
        """
        pass
    
    @abstractmethod
    def close(self):
        """Closes the environment."""
        pass

class UnityAdapter(BaseEnv):
    """Adapter for Unity ML-Agents Environment."""
    
    def __init__(self, file_name, no_graphics=True, seed=0, worker_id=0):
        """Initialize the Unity Environment.
        
        Args:
            file_name (str): Path to the Unity environment binary.
            no_graphics (bool): Whether to run without graphics.
            seed (int): Random seed (not always used by UnityEnv directly in init but good practice).
            worker_id (int): Port to use for communication.
        """
        self.env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics, seed=seed, worker_id=worker_id)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        
    def reset(self):
        """Resets the environment."""
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations[0]
        
    def step(self, action):
        """Takes a step in the environment."""
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return next_state, reward, done, None
        
    def close(self):
        """Closes the environment."""
        self.env.close()
