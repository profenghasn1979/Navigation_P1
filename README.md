# Navigation Project (DQN)

This project implements a Deep Q-Network (DQN) agent to solve the Unity ML-Agents "Banana" Navigation environment.

## Environment Details

*   **Goal**: The agent must navigate a large, square world to collect yellow bananas and avoid blue bananas.
*   **Rewards**:
    *   +1 for collecting a yellow banana.
    *   -1 for collecting a blue banana.
*   **State Space**: The state space has **37 dimensions**, including the agent's velocity and ray-based perception of objects around the agent's forward direction.
*   **Action Space**: **4 discrete actions** are available:
    *   `0`: Move forward.
    *   `1`: Move backward.
    *   `2`: Turn left.
    *   `3`: Turn right.
*   **Solved Condition**: The environment is considered solved when the agent achieves an average score of **+13 over 100 consecutive episodes**.

## Project Structure

The project follows a modular architecture using several design patterns:

- **src/config.py**: Configuration Pattern. Centralized hyperparameters.
- **src/env_adapter.py**: Adapter Pattern. Wraps the Unity environment to expose a standard API.
- **src/agent.py**: Strategy Pattern. Implements the Agent logic (DQN).
- **src/model.py**: Factory Pattern. Creates the Q-Network.
- **src/buffer.py**: Replay Buffer implementation.
- **train.py**: Main training script.
- **evaluate.py**: Evaluation script.

## Requirements

- Python 3.6+
- PyTorch
- UnityAgents
- NumPy
- Matplotlib

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install torch numpy unityagents matplotlib
   ```
3. Download the Unity Environment "Banana" for your OS and place it in a known directory.

## Usage

### Training

1. **Activate the Virtual Environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Run Training**:
   ```bash
   python train.py --env Banana.app --worker_id 1
   ```
   *Note: If you encounter permission issues, run `chmod -R 755 Banana.app` and `xattr -cr Banana.app` first.*
   *Note: If you get a `UnityTimeOutException`, try changing the `worker_id` (e.g., `--worker_id 2`).*

   The script will save the best model weights to `checkpoint.pth` once the environment is solved (avg score >= 13.0).

### Evaluation

To watch a trained agent:

```bash
source venv/bin/activate
python evaluate.py --env Banana.app --checkpoint checkpoint.pth
```

## Hyperparameters

You can modify hyperparameters in `src/config.py`. Key parameters include:
- `BUFFER_SIZE`: 100,000
- `BATCH_SIZE`: 64
- `GAMMA`: 0.99
- `LR`: 5e-4
- `UPDATE_EVERY`: 4
