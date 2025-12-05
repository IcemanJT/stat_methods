import numpy as np
import matplotlib.pyplot as plt
from server_markov import (
    step_100_users_const,
    step_100_users_state_dependent,
    simulate_100_users,
    empirical_distribution_from_states
)

# ===========================
# ZADANIE D
# ===========================

N_users = 100
N_steps = 10_000
x0 = 0

rng = np.random.default_rng(789)

# 1) Rozkład dla stałych prawdopodobieństw (jak w C)
states_const = simulate_100_users(
    step_func=step_100_users_const,
    x0=x0,
    N_steps=N_steps,
    rng=rng,
    N_users=N_users,
    p_login=0.2,
    p_stay_logged_in=0.5
)
dist_const = empirical_distribution_from_states(states_const, N_users)

# 2) Rozkład dla logout zależnego od x
states_sd = simulate_100_users(
    step_func=step_100_users_state_dependent,
    x0=x0,
    N_steps=N_steps,
    rng=rng,
    N_users=N_users,
    p_login=0.2
)
dist_sd = empirical_distribution_from_states(states_sd, N_users)

print("Suma rozkładu (const):", dist_const.sum())
print("Suma rozkładu (state-dependent):", dist_sd.sum())

# Porównawczy wykres
x_vals = np.arange(N_users + 1)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, dist_const, label="Stałe P(logout)=0.5 (Zad. C)")
plt.plot(x_vals, dist_sd, label="P(stay|x)=0.008x+0.1 (Zad. D)")
plt.grid(True)
plt.xlabel("Liczba zalogowanych użytkowników")
plt.ylabel("Prawdopodobieństwo empiryczne")
plt.title("Porównanie rozkładów – stałe vs zależne od x")
plt.legend()
plt.show()
