import numpy as np
import matplotlib.pyplot as plt
from server_markov import (
    step_100_users_const,
    simulate_100_users,
    empirical_distribution_from_states
)

# ===========================
# ZADANIE C
# ===========================

N_users = 100
rng = np.random.default_rng(456)
N_steps = 10_000
x0 = 0

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
print("Suma rozkładu (stałe prawdopodobieństwa):", dist_const.sum())

plt.figure(figsize=(10, 5))
plt.bar(np.arange(N_users + 1), dist_const)
plt.grid(True, axis="y")
plt.xlabel("Liczba zalogowanych użytkowników")
plt.ylabel("Empiryczne prawdopodobieństwo")
plt.title("Zadanie C – rozkład liczby zalogowanych użytkowników (stałe prawdopodobieństwa)")
plt.show()

# Można zapisać do pliku, jeśli chcesz używać poza importem:
np.save("dist_const_C.npy", dist_const)
