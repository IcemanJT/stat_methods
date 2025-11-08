"""
Task C – Number of Rounds (L) Until the Game Ends

Given:
- a = 50
- b = 50
- pA ∈ {1/5, 1/2, 4/5}

For each value of pA, find:
- The distribution P(L)
- The value of L
- The average duration (expected number of rounds).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from gambler_ruin import simulate_multiple_games, theoretical_expected_rounds

# Given parameters
a = 50
b = 50
pA_values = [1/5, 1/2, 4/5]  # [0.2, 0.5, 0.8]
num_simulations = 10000

print("=" * 70)
print("Task C – Number of Rounds (L) Until the Game Ends")
print("=" * 70)
print(f"Given: a = {a}, b = {b}")
print(f"Number of simulations: {num_simulations}")
print()

# Create figure for distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Distribution of Game Duration L for Different pA Values', fontsize=14)

for idx, pA in enumerate(pA_values):
    print(f"\nFor pA = {pA} ({pA:.1f}):")
    print("-" * 70)
    
    # Simulation
    results = simulate_multiple_games(a, b, pA, num_simulations)
    num_rounds_list = results['num_rounds']
    
    # Distribution P(L)
    rounds_counter = Counter(num_rounds_list)
    total = len(num_rounds_list)
    max_rounds = max(num_rounds_list)
    
    # Create distribution
    L_values = sorted(rounds_counter.keys())
    P_L = [rounds_counter[L] / total for L in L_values]
    
    # Statistics
    avg_rounds_sim = results['avg_rounds']
    median_rounds = np.median(num_rounds_list)
    std_rounds = np.std(num_rounds_list)
    
    # Theoretical expected value (approximation)
    # Note: Exact theoretical formula for distribution is complex
    # We'll compute expected value
    try:
        avg_rounds_theory = theoretical_expected_rounds(a, b, pA)
    except:
        avg_rounds_theory = None
    
    print(f"  Average duration (simulation) = {avg_rounds_sim:.2f} rounds")
    if avg_rounds_theory is not None:
        print(f"  Average duration (theoretical) = {avg_rounds_theory:.2f} rounds")
    print(f"  Median duration = {median_rounds:.2f} rounds")
    print(f"  Standard deviation = {std_rounds:.2f} rounds")
    print(f"  Minimum rounds = {min(num_rounds_list)}")
    print(f"  Maximum rounds = {max(num_rounds_list)}")
    
    # Plot distribution
    ax = axes[idx]
    # Use histogram for better visualization
    ax.hist(num_rounds_list, bins=min(100, max_rounds//10 + 1), density=True, alpha=0.7, edgecolor='black')
    ax.axvline(avg_rounds_sim, color='r', linestyle='--', linewidth=2, label=f'Mean = {avg_rounds_sim:.1f}')
    if avg_rounds_theory is not None:
        ax.axvline(avg_rounds_theory, color='g', linestyle='--', linewidth=2, label=f'Theory = {avg_rounds_theory:.1f}')
    ax.set_xlabel('Number of Rounds (L)', fontsize=10)
    ax.set_ylabel('Probability Density P(L)', fontsize=10)
    ax.set_title(f'pA = {pA:.1f}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Print some distribution statistics
    print(f"  Distribution range: L ∈ [{min(L_values)}, {max(L_values)}]")
    print(f"  Most common L value: {rounds_counter.most_common(1)[0][0]} (occurred {rounds_counter.most_common(1)[0][1]} times)")

plt.tight_layout()
# Save to plots/tasks1/ directory (relative to repo root)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
plot_path = os.path.join(repo_root, 'plots', 'tasks1', 'task_c_distributions.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nDistribution plots saved to '{plot_path}'")

print("\n" + "=" * 70)

