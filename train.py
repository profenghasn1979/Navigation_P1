import argparse
import time
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from src.config import Config
from src.env_adapter import UnityAdapter
from src.agent import DQNAgent

def train(env_path, n_episodes=2000, max_t=1000, worker_id=0):
    """Deep Q-Learning.
    
    Args:
        env_path (str): Path to the Unity environment binary.
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        worker_id (int): Port to use for communication.
    """
    # Initialize Config
    config = Config()
    
    # Initialize Environment
    env = UnityAdapter(file_name=env_path, no_graphics=True, seed=config.SEED, worker_id=worker_id)
    
    # Initialize Agent
    agent = DQNAgent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, seed=config.SEED, config=config)
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = config.EPS_START             # initialize epsilon
    
    
    print(f"Training started with device: {torch.device('cpu')}")
    
    start_time = time.time()
    for i_episode in range(1, n_episodes+1):
        episode_start = time.time()
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(config.EPS_END, config.EPS_DECAY*eps) # decrease epsilon
        
        duration = time.time() - episode_start
        print('\rEpisode {}\tAvg Score: {:.2f}\tDuration: {:.2f}s'.format(i_episode, np.mean(scores_window), duration), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAvg Score: {:.2f}\tDuration: {:.2f}s'.format(i_episode, np.mean(scores_window), duration))
        if np.mean(scores_window)>=13.0:
            total_time = time.time() - start_time
            print('\nEnvironment solved in {:d} episodes!\tAvg Score: {:.2f}\tTotal Time: {:.2f}s'.format(i_episode-100, np.mean(scores_window), total_time))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
            
    env.close()
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN Agent for Banana Navigation')
    parser.add_argument('--env', type=str, required=True, help='Path to the Unity Environment binary')
    parser.add_argument('--worker_id', type=int, default=0, help='Worker ID for Unity Environment (default: 0)')
    args = parser.parse_args()
    
    train(args.env, worker_id=args.worker_id)
