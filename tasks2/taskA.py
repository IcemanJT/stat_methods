import numpy as np
import matplotlib.pyplot as plt
from server_markov import build_transition_matrix_two_users, power_iteration

# ===========================
# ZADANIE A
# ===========================

# Budowa macierzy P dla 2 użytkowników
P = build_transition_matrix_two_users(p_login=0.2, p_stay_logged_in=0.5)
print("Macierz przejścia P:")
print(P)
print("Suma wierszy:", P.sum(axis=1))

# Iterowanie P^N
P_powers, P_lim, converged_N = power_iteration(P, N_max=500, eps=1e-6)

if converged_N is not None:
    print(f"Conwergencja wykryta przy N = {converged_N}")
else:
    print("Brak pełnej konwergencji w zadanym N_max.")

print("\nMacierz graniczna (P^N dla dużego N):")
print(P_lim)

pi_approx = P_lim[0]
print("\nPrzybliżony rozkład stacjonarny:", pi_approx)

# Dokładny rozkład stacjonarny (wyliczony analitycznie)
pi_exact = np.array([25/49, 20/49, 4/49])
print("Dokładny rozkład stacjonarny:", pi_exact)

# Przygotowanie danych do wykresu
P_arr = np.stack(P_powers)    # shape: (N+1, 3, 3)
Ns = np.arange(len(P_powers))
num_states = P.shape[0]

fig, axes = plt.subplots(num_states, num_states, figsize=(12, 9), sharex=True)
fig.suptitle("Konwergencja elementów P^N do rozkładu stacjonarnego")

for i in range(num_states):
    for j in range(num_states):
        ax = axes[i, j]
        ax.plot(Ns, P_arr[:, i, j])
        ax.axhline(pi_exact[j], linestyle='--')
        ax.grid(True)
        if i == num_states - 1:
            ax.set_xlabel("N")
        if j == 0:
            ax.set_ylabel(f"P[{i},{j}]")

plt.tight_layout()
plt.show()
