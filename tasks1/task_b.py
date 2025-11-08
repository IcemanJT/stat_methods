"""
Task B – Gambler's Ruin (Two Players A and B)

Given:
- a + b = 100
- pA = 0.5

Find:
- P(ruin of A)
- The value of a (initial capital of A) for which the probability is evaluated
- Comparison with the theoretical result.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from gambler_ruin import simulate_multiple_games, theoretical_P_ruin_A

# Given parameters
total_capital = 100
pA = 0.5
num_simulations = 10000

# Test different values of a (from 10 to 90 in steps of 10)
a_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]

print("=" * 70)
print("Task B – Gambler's Ruin")
print("=" * 70)
print(f"Given: a + b = {total_capital}, pA = {pA}")
print(f"Number of simulations: {num_simulations}")
print()

results_table = []

for a in a_values:
    b = total_capital - a
    
    # Simulation
    results = simulate_multiple_games(a, b, pA, num_simulations)
    P_ruin_A_sim = results['P_ruin_A']
    
    # Theoretical (for pA = 0.5: P_ruin_A = b / (a + b))
    P_ruin_A_theory = theoretical_P_ruin_A(a, b, pA)
    
    # Comparison
    error = abs(P_ruin_A_sim - P_ruin_A_theory)
    relative_error = (error / P_ruin_A_theory * 100) if P_ruin_A_theory > 0 else 0
    
    results_table.append({
        'a': a,
        'b': b,
        'P_ruin_A_sim': P_ruin_A_sim,
        'P_ruin_A_theory': P_ruin_A_theory,
        'error': error,
        'relative_error': relative_error
    })
    
    print(f"a = {a:2d}, b = {b:2d}:")
    print(f"  Simulation P(ruin of A) = {P_ruin_A_sim:.6f}")
    print(f"  Theoretical P(ruin of A) = {P_ruin_A_theory:.6f}")
    print(f"  Absolute error = {error:.6f} ({relative_error:.2f}%)")
    print()

print("=" * 70)
print("Summary Table:")
print("=" * 70)
print(f"{'a':>4} {'b':>4} {'P_ruin_A (sim)':>18} {'P_ruin_A (theory)':>20} {'Error':>12} {'Rel Error %':>12}")
print("-" * 70)
for r in results_table:
    print(f"{r['a']:>4} {r['b']:>4} {r['P_ruin_A_sim']:>18.6f} {r['P_ruin_A_theory']:>20.6f} {r['error']:>12.6f} {r['relative_error']:>12.2f}")

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Task B: P(ruin of A) vs Initial Capital a (pA = {pA}, a + b = {total_capital})', fontsize=14)

# Extract data for plotting
a_vals = [r['a'] for r in results_table]
sim_vals = [r['P_ruin_A_sim'] for r in results_table]
theory_vals = [r['P_ruin_A_theory'] for r in results_table]
error_vals = [r['error'] for r in results_table]

# Plot 1: Simulation vs Theoretical
ax1 = axes[0]
ax1.plot(a_vals, sim_vals, 'o-', linewidth=2, markersize=8, label='Simulation', color='blue')
ax1.plot(a_vals, theory_vals, 's--', linewidth=2, markersize=8, label='Theoretical', color='red')
ax1.set_xlabel('Initial Capital a', fontsize=11)
ax1.set_ylabel('P(ruin of A)', fontsize=11)
ax1.set_title('Simulation vs Theoretical', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Error
ax2 = axes[1]
ax2.plot(a_vals, error_vals, 'o-', linewidth=2, markersize=8, color='green')
ax2.set_xlabel('Initial Capital a', fontsize=11)
ax2.set_ylabel('Absolute Error', fontsize=11)
ax2.set_title('Simulation Error', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# Save to plots/tasks1/ directory (relative to repo root)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
plot_path = os.path.join(repo_root, 'plots', 'tasks1', 'task_b_comparison.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nPlots saved to '{plot_path}'")

print("\n" + "=" * 70)
print("Note: For pA = 0.5, theoretical formula is P(ruin of A) = b / (a + b)")
print("=" * 70)

