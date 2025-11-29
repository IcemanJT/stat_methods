import numpy as np
import matplotlib.pyplot as plt

# ===========================================
# ZADANIE B – Monte Carlo simulation for P
# ===========================================

P = np.array([
    [0.64, 0.32, 0.04],
    [0.40, 0.50, 0.10],
    [0.25, 0.50, 0.25]
])

pi_exact = np.array([25/49, 20/49, 4/49])
num_states = 3

rng = np.random.default_rng(123)

def simulate_chain(P, x0, N, rng):
    states = np.zeros(N + 1, dtype=int)
    states[0] = x0
    cur = x0
    for t in range(1, N + 1):
        cur = rng.choice(num_states, p=P[cur])
        states[t] = cur
    return states

N_max = 10000

all_results = {}

for x0 in range(num_states):
    visited = simulate_chain(P, x0, N_max, rng)

    counts = np.zeros((N_max + 1, num_states))
    running = np.zeros(num_states)

    for t in range(N_max + 1):
        running[visited[t]] += 1
        if t > 0:
            counts[t] = running / t

    all_results[x0] = counts

# Plot empirical convergence
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle("Monte Carlo – zbieżność do rozkładu stacjonarnego")

Ns = np.arange(N_max + 1)

for x0 in range(num_states):
    ax = axes[x0]
    counts = all_results[x0]

    for s in range(num_states):
        ax.plot(Ns[1:], counts[1:, s], label=f"Stan {s}")
        ax.axhline(pi_exact[s], linestyle="--")

    ax.set_title(f"Start z {x0}")
    ax.set_xlabel("N")
    ax.grid(True)

axes[0].set_ylabel("Prawdopodobieństwo empiryczne")
plt.legend()
plt.tight_layout()
plt.show()
