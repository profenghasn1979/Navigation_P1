from dataclasses import dataclass

@dataclass
class Config:
    """
    Configuration for the DQN Agent and Training Process.
    """
    # Environment
    STATE_SIZE: int = 37
    ACTION_SIZE: int = 4
    
    # Training Hyperparameters
    BUFFER_SIZE: int = 100000  # replay buffer size
    BATCH_SIZE: int = 64       # minibatch size
    GAMMA: float = 0.99        # discount factor
    TAU: float = 1e-3          # for soft update of target parameters
    LR: float = 5e-4           # learning rate 
    UPDATE_EVERY: int = 4      # how often to update the network
    
    # Epsilon Greedy Strategy
    EPS_START: float = 1.0
    EPS_END: float = 0.01
    EPS_DECAY: float = 0.995
    
    # Model Architecture
    HIDDEN_LAYERS: tuple = (64, 64)
    
    # Double DQN
    USE_DOUBLE_DQN: bool = True
    
    # Seed
    SEED: int = 0
