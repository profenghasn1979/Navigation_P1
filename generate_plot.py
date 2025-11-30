import matplotlib.pyplot as plt
import numpy as np

# Data from training log
episodes = [100, 200, 300, 400, 500, 505]
scores = [0.58, 4.08, 7.46, 11.24, 12.80, 13.01]

plt.figure(figsize=(10, 6))
plt.plot(episodes, scores, marker='o', linestyle='-')
plt.title('Average Score over 100 Consecutive Episodes')
plt.xlabel('Episode #')
plt.ylabel('Average Score')
plt.axhline(y=13.0, color='r', linestyle='--', label='Solved Threshold (+13)')
plt.legend()
plt.grid(True)
plt.savefig('scores_plot.png')
print("Plot saved to scores_plot.png")
