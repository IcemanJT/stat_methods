import numpy as np
import matplotlib.pyplot as plt

# ===========================================
# ZADANIE C – 100 users, constant probabilities
# ===========================================

N_users = 100
login_p = 0.2
stay_p_logged_in = 0.5

rng = np.random.default_rng(456)

def step_const(x, rng):
    stays = rng.binomial(x, stay_p_logged_in)
    logins = rng.binomial(N_users - x, login_p)
    return stays + logins

def simulate_const(x0, N, rng):
    states = np.zeros(N + 1, dtype=int)
    states[0] = x0
    x = x0
    for t in range(1, N + 1):
        x = step_const(x, rng)
        states[t] = x
    return states

N_steps = 10000
x0 = 0

states = simulate_const(x0, N_steps, rng)
counts = np.bincount(states, minlength=N_users + 1)
dist = counts / counts.sum()

print("Suma rozkładu:", dist.sum())

plt.figure(figsize=(10,5))
plt.bar(np.arange(N_users + 1), dist)
plt.grid(True, axis="y")
plt.xlabel("Liczba zalogowanych użytkowników")
plt.ylabel("Empiryczne prawdopodobieństwo")
plt.title("Zadanie C – Rozkład liczby zalogowanych użytkowników")
plt.show()

np.save("dist_C.npy", dist)   # zapisuje rozkład do pliku
