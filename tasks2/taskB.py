import numpy as np
import matplotlib.pyplot as plt
from server_markov import (
    build_transition_matrix_two_users,
    simulate_markov_chain,
    empirical_probabilities_from_path
)

# ===========================
# ZADANIE B
# ===========================

P = build_transition_matrix_two_users(p_login=0.2, p_stay_logged_in=0.5)
num_states = P.shape[0]

pi_exact = np.array([25/49, 20/49, 4/49])

rng = np.random.default_rng(123)
N_max = 10_000

all_counts = {}

for x0 in range(num_states):
    states = simulate_markov_chain(P, x0, N_max, rng)
    counts = empirical_probabilities_from_path(states, num_states)
    all_counts[x0] = counts

Ns = np.arange(N_max + 1)

fig, axes = plt.subplots(1, num_states, figsize=(15, 5), sharey=True)
fig.suptitle("Zbieżność empiryczna do rozkładu stacjonarnego (Monte Carlo)")

for x0 in range(num_states):
    ax = axes[x0]
    counts = all_counts[x0]

    for s in range(num_states):
        ax.plot(Ns[1:], counts[1:, s], label=f"stan {s}")
        ax.axhline(pi_exact[s], linestyle='--')

    ax.set_title(f"Start ze stanu {x0}")
    ax.set_xlabel("N")
    ax.grid(True)

axes[0].set_ylabel("Prawdopodobieństwo empiryczne")
axes[-1].legend(loc="best")
plt.tight_layout()
plt.show()
