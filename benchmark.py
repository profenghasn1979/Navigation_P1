import torch
import time
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_layers=(64, 64)):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, hidden_layers[0]))
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.layers.append(nn.Linear(hidden_layers[-1], action_size))

    def forward(self, state):
        x = state
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)

def benchmark(device_name):
    device = torch.device(device_name)
    model = QNetwork(37, 4, 0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # Dummy data
    states = torch.randn(64, 37).to(device)
    
    start = time.time()
    for _ in range(1000):
        # Forward
        output = model(states)
        # Backward
        loss = output.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    end = time.time()
    print(f"{device_name}: {end - start:.4f} seconds for 1000 steps")

if __name__ == "__main__":
    print("Benchmarking CPU...")
    benchmark("cpu")
    
    if torch.backends.mps.is_available():
        print("Benchmarking MPS...")
        benchmark("mps")
    else:
        print("MPS not available.")
