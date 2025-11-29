import numpy as np
import matplotlib.pyplot as plt

# ===========================================
# ZADANIE A – Transition matrix, P^N, convergence
# ===========================================

# Transition matrix for states 0,1,2
P = np.array([
    [0.64, 0.32, 0.04],
    [0.40, 0.50, 0.10],
    [0.25, 0.50, 0.25]
])

num_states = 3

print("Macierz przejścia P:")
print(P)
print("Suma wierszy:", P.sum(axis=1))

# Power iteration: compute P^N until convergence
N_max = 500
eps = 1e-6

P_prev = np.eye(num_states)
P_powers = [P_prev]
converged_N = None

for N in range(1, N_max + 1):
    P_curr = P_prev @ P
    P_powers.append(P_curr)

    diff = np.max(np.abs(P_curr - P_prev))

    if diff < eps and converged_N is None:
        converged_N = N
        print(f"Conwergencja przy N = {N}, różnica = {diff:.2e}")

    P_prev = P_curr

if converged_N is None:
    print("UWAGA: brak pełnej konwergencji do N = 500.")

P_lim = P_powers[-1]
print("\nMacierz graniczna P^inf:")
print(P_lim)

# Approximated stationary distribution
pi_approx = P_lim[0]
print("\nPrzybliżona rozkład stacjonarny:", pi_approx)

# Exact stationary distribution:
pi_exact = np.array([25/49, 20/49, 4/49])
print("Dokładny:", pi_exact)

# Plot convergence of each entry P_ij(N)
P_arr = np.stack(P_powers)
Ns = np.arange(len(P_powers))

fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
fig.suptitle("Konwergencja elementów macierzy P^N")

for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        ax.plot(Ns, P_arr[:, i, j], label=f"P^{N}[{i},{j}]")
        ax.axhline(pi_exact[j], linestyle="--")
        ax.grid(True)
        if i == 2:
            ax.set_xlabel("N")
        if j == 0:
            ax.set_ylabel(f"P[{i},{j}]")

plt.tight_layout()
plt.show()
