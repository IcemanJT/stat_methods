"""
Task A – Gambler's Ruin (Two Players A and B)

Given:
- a = 50
- b = 50

Find:
- pA (probability that player A wins a single round)
- P(ruin of A) (probability that player A goes bankrupt)
- Comparison of the simulation result with the analytical (theoretical) result.
"""

import matplotlib.pyplot as plt
import os
from gambler_ruin import simulate_multiple_games, theoretical_P_ruin_A

# Given parameters
a = 50
b = 50

# We need to find pA - let's test different values
# For fair comparison, we'll test pA values and see which gives interesting results
# Typically, we'd want to test pA < 0.5, pA = 0.5, and pA > 0.5

pA_values = [0.3, 0.4, 0.5, 0.6, 0.7]
num_simulations = 10000

print("=" * 70)
print("Task A – Gambler's Ruin")
print("=" * 70)
print(f"Given: a = {a}, b = {b}")
print(f"Number of simulations: {num_simulations}")
print()

# Store results for plotting
sim_results = []
theory_results = []
errors = []
relative_errors = []

for pA in pA_values:
    print(f"\nFor pA = {pA}:")
    print("-" * 70)
    
    # Simulation
    results = simulate_multiple_games(a, b, pA, num_simulations)
    P_ruin_A_sim = results['P_ruin_A']
    
    # Theoretical
    P_ruin_A_theory = theoretical_P_ruin_A(a, b, pA)
    
    # Comparison
    error = abs(P_ruin_A_sim - P_ruin_A_theory)
    relative_error = (error / P_ruin_A_theory * 100) if P_ruin_A_theory > 0 else 0
    
    # Store for plotting
    sim_results.append(P_ruin_A_sim)
    theory_results.append(P_ruin_A_theory)
    errors.append(error)
    relative_errors.append(relative_error)
    
    print(f"  Simulation P(ruin of A) = {P_ruin_A_sim:.6f}")
    print(f"  Theoretical P(ruin of A) = {P_ruin_A_theory:.6f}")
    print(f"  Absolute error = {error:.6f}")
    print(f"  Relative error = {relative_error:.2f}%")
    print(f"  Average number of rounds = {results['avg_rounds']:.2f}")

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Task A: P(ruin of A) vs pA (a = {a}, b = {b})', fontsize=14)

# Plot 1: Simulation vs Theoretical
ax1 = axes[0]
ax1.plot(pA_values, sim_results, 'o-', linewidth=2, markersize=8, label='Simulation', color='blue')
ax1.plot(pA_values, theory_results, 's--', linewidth=2, markersize=8, label='Theoretical', color='red')
ax1.set_xlabel('pA (Probability A wins)', fontsize=11)
ax1.set_ylabel('P(ruin of A)', fontsize=11)
ax1.set_title('Simulation vs Theoretical', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Error
ax2 = axes[1]
ax2.plot(pA_values, errors, 'o-', linewidth=2, markersize=8, color='green')
ax2.set_xlabel('pA (Probability A wins)', fontsize=11)
ax2.set_ylabel('Absolute Error', fontsize=11)
ax2.set_title('Simulation Error', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# Save to plots/tasks1/ directory (relative to repo root)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
plot_path = os.path.join(repo_root, 'plots', 'tasks1', 'task_a_comparison.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nPlots saved to '{plot_path}'")

print("\n" + "=" * 70)
print("Note: For a = b = 50, when pA = 0.5, the game is fair and")
print("P(ruin of A) = b/(a+b) = 0.5 (theoretical)")
print("=" * 70)

