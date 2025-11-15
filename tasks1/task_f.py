"""
Task F – Capital (k) of Player A After N Rounds

Given:
- a = 50
- b = 50
- pA = 0.2
- N ∈ {1, 10, 50, 60, 70, 80}

For each N, determine:
- The capital k of player A after N rounds
- The probability distribution P(k)
  (Note: many simulations are needed to estimate the probabilities.)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from gambler_ruin import simulate_capital_after_N_rounds

# Given parameters
a = 50
b = 50
pA = 0.2
N_values = [1, 10, 50, 60, 70, 80]
num_simulations = 50000  # Many simulations needed for accurate probability estimates

print("=" * 70)
print("Task F – Capital (k) of Player A After N Rounds")
print("=" * 70)
print(f"Given: a = {a}, b = {b}, pA = {pA}")
print(f"Number of simulations: {num_simulations}")
print()

# Create figure for distributions
num_plots = len(N_values)
cols = 3
rows = (num_plots + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
if rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()
fig.suptitle(f'Distribution of Capital k After N Rounds (pA = {pA})', fontsize=14)

for idx, N in enumerate(N_values):
    print(f"\nFor N = {N}:")
    print("-" * 70)
    
    # Simulate capital after N rounds
    final_capitals = simulate_capital_after_N_rounds(a, b, pA, N, num_simulations)
    
    # Distribution P(k)
    capital_counter = Counter(final_capitals)
    total = len(final_capitals)
    
    k_values = sorted(capital_counter.keys())
    P_k = [capital_counter[k] / total for k in k_values]
    
    # Statistics
    mean_capital = np.mean(final_capitals)
    median_capital = np.median(final_capitals)
    std_capital = np.std(final_capitals)
    min_capital = min(final_capitals)
    max_capital = max(final_capitals)
    
    print(f"  Mean capital k = {mean_capital:.2f}")
    print(f"  Median capital k = {median_capital:.2f}")
    print(f"  Standard deviation = {std_capital:.2f}")
    print(f"  Range: k ∈ [{min_capital}, {max_capital}]")
    print(f"  Most common k value: {capital_counter.most_common(1)[0][0]} (P = {capital_counter.most_common(1)[0][1]/total:.4f})")
    
    # Plot distribution
    ax = axes[idx]
    ax.bar(k_values, P_k, width=0.8, edgecolor='black')
    ax.set_xlabel('Capital k')
    ax.set_ylabel('P(k)')
    ax.axvline(mean_capital, color='r', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_capital:.1f}')
    ax.axvline(a, color='g', linestyle='--', linewidth=2, 
               label=f'Initial = {a}')
    ax.set_title(f'N = {N}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Print top 5 most probable k values
    top_5 = capital_counter.most_common(5)
    print(f"  Top 5 most probable k values:")
    for k, count in top_5:
        print(f"    k = {k:3d}: P(k) = {count/total:.4f}")

# Hide unused subplots
for idx in range(len(N_values), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
# Save to plots/tasks1/ directory (relative to repo root)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
plot_path = os.path.join(repo_root, 'plots', 'tasks1', 'task_f_distributions.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nDistribution plots saved to '{plot_path}'")

print("\n" + "=" * 70)
print("Note: For pA = 0.2, player A is at a disadvantage.")
print("As N increases, the distribution shifts toward lower capital values.")
print("=" * 70)

