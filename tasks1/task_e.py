"""
Task E – Maximum Game Duration (LMAX)

Given:
- a = 50
- b = 50

Find:
- LMAX (the maximum observed number of rounds) across 1000 simulated games.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from gambler_ruin import simulate_multiple_games

# Given parameters
a = 50
b = 50
num_simulations = 1000

# Test different pA values to see how LMAX varies
pA_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

print("=" * 70)
print("Task E – Maximum Game Duration (LMAX)")
print("=" * 70)
print(f"Given: a = {a}, b = {b}")
print(f"Number of simulations: {num_simulations}")
print()

results_table = []

for pA in pA_values:
    results = simulate_multiple_games(a, b, pA, num_simulations)
    num_rounds_list = results['num_rounds']
    
    LMAX = max(num_rounds_list)
    LMIN = min(num_rounds_list)
    avg_rounds = results['avg_rounds']
    
    results_table.append({
        'pA': pA,
        'LMAX': LMAX,
        'LMIN': LMIN,
        'avg_rounds': avg_rounds
    })
    
    print(f"pA = {pA:.1f}:")
    print(f"  LMAX = {LMAX} rounds")
    print(f"  LMIN = {LMIN} rounds")
    print(f"  Average = {avg_rounds:.2f} rounds")
    print()

print("=" * 70)
print("Summary Table:")
print("=" * 70)
print(f"{'pA':>6} {'LMAX':>10} {'LMIN':>10} {'Average':>12}")
print("-" * 70)
for r in results_table:
    print(f"{r['pA']:>6.1f} {r['LMAX']:>10} {r['LMIN']:>10} {r['avg_rounds']:>12.2f}")

# Create plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Task E: Game Duration Statistics vs pA (a = {a}, b = {b})', fontsize=14)

# Extract data for plotting
pA_vals = [r['pA'] for r in results_table]
lmax_vals = [r['LMAX'] for r in results_table]
lmin_vals = [r['LMIN'] for r in results_table]
avg_vals = [r['avg_rounds'] for r in results_table]

# Plot 1: LMAX
ax1 = axes[0]
ax1.plot(pA_vals, lmax_vals, 'o-', linewidth=2, markersize=8, color='red')
ax1.set_xlabel('pA (Probability A wins)', fontsize=11)
ax1.set_ylabel('LMAX (Maximum Rounds)', fontsize=11)
ax1.set_title('Maximum Game Duration', fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: LMIN
ax2 = axes[1]
ax2.plot(pA_vals, lmin_vals, 'o-', linewidth=2, markersize=8, color='blue')
ax2.set_xlabel('pA (Probability A wins)', fontsize=11)
ax2.set_ylabel('LMIN (Minimum Rounds)', fontsize=11)
ax2.set_title('Minimum Game Duration', fontsize=12)
ax2.grid(True, alpha=0.3)

# Plot 3: Average
ax3 = axes[2]
ax3.plot(pA_vals, avg_vals, 'o-', linewidth=2, markersize=8, color='green')
ax3.set_xlabel('pA (Probability A wins)', fontsize=11)
ax3.set_ylabel('Average Rounds', fontsize=11)
ax3.set_title('Average Game Duration', fontsize=12)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
# Save to plots/tasks1/ directory (relative to repo root)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
plot_path = os.path.join(repo_root, 'plots', 'tasks1', 'task_e_duration.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nPlots saved to '{plot_path}'")

print("\n" + "=" * 70)
print(f"Note: LMAX is the maximum observed duration across {num_simulations} games.")
print("For pA = 0.5 (fair game), games can be very long.")
print("=" * 70)

