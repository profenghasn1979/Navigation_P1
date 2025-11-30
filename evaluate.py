import argparse
import torch
import numpy as np
from src.config import Config
from src.env_adapter import UnityAdapter
from src.agent import DQNAgent

def evaluate(env_path, checkpoint_path='checkpoint.pth', n_episodes=5):
    """Evaluate a trained agent.
    
    Args:
        env_path (str): Path to the Unity environment binary.
        checkpoint_path (str): Path to the saved model weights.
        n_episodes (int): Number of episodes to run.
    """
    # Initialize Config
    config = Config()
    
    # Initialize Environment (with graphics this time if possible, but usually headless on servers)
    # Note: If running locally with a screen, you might want no_graphics=False
    env = UnityAdapter(file_name=env_path, no_graphics=False, seed=config.SEED)
    
    # Initialize Agent
    agent = DQNAgent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, seed=config.SEED, config=config)
    
    # Load weights
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
    
    for i in range(n_episodes):
        state = env.reset()
        score = 0
        while True:
            action = agent.act(state, eps=0.0) # Greedy action
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if done:
                break
        print(f"Episode {i+1}: Score = {score}")
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DQN Agent for Banana Navigation')
    parser.add_argument('--env', type=str, required=True, help='Path to the Unity Environment binary')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Path to the checkpoint file')
    args = parser.parse_args()
    
    evaluate(args.env, args.checkpoint)
