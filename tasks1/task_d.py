"""
Task D – Trajectory of the Number of Wins for One Player (A or B)

Given:
- a = 10
- a + b = 20
- pA ∈ {1/5, 1/2, 4/5}

For each value of pA, find:
- The number of wins over time for the selected player
- Plot trajectories for 3 separate games on one common graph.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from gambler_ruin import simulate_game_with_wins

# Given parameters
a = 10
b = 20 - a  # a + b = 20
pA_values = [1/5, 1/2, 4/5]  # [0.2, 0.5, 0.8]
num_games = 3

print("=" * 70)
print("Task D – Trajectory of the Number of Wins for Player A")
print("=" * 70)
print(f"Given: a = {a}, a + b = {a + b} (so b = {b})")
print(f"Number of games to plot: {num_games}")
print()

# Create figure with subplots for each pA value
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Trajectories of Number of Wins for Player A (3 games per pA)', fontsize=14)

for idx, pA in enumerate(pA_values):
    print(f"\nFor pA = {pA} ({pA:.1f}):")
    print("-" * 70)
    
    ax = axes[idx]
    
    # Simulate 3 games
    for game_num in range(num_games):
        A_wins, total_rounds, capital_history, wins_history = simulate_game_with_wins(a, b, pA)
        
        # Plot trajectory
        rounds_list = list(range(len(wins_history)))
        ax.plot(rounds_list, wins_history, linewidth=2, alpha=0.7, 
                label=f'Game {game_num + 1} ({total_rounds} rounds, A {"won" if A_wins else "lost"})')
        
        final_wins = wins_history[-1] if wins_history else 0
        print(f"  Game {game_num + 1}: {total_rounds} rounds, A {'won' if A_wins else 'lost'}, Final wins: {final_wins}")
    
    ax.set_xlabel('Round Number', fontsize=10)
    ax.set_ylabel('Cumulative Wins for Player A', fontsize=10)
    ax.set_title(f'pA = {pA:.1f}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
# Save to plots/tasks1/ directory (relative to repo root)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
plot_path = os.path.join(repo_root, 'plots', 'tasks1', 'task_d_trajectories.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nTrajectory plots saved to '{plot_path}'")

print("\n" + "=" * 70)

