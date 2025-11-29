import numpy as np
import matplotlib.pyplot as plt

dist_const = np.load("dist_C.npy")

# ===========================================
# ZADANIE D – State-dependent logout probability
# ===========================================

N_users = 100
login_p = 0.2

rng = np.random.default_rng(789)

def step_state_dep(x, rng):
    p_stay = 0.008 * x + 0.1
    p_stay = np.clip(p_stay, 0, 1)

    stays = rng.binomial(x, p_stay)
    logins = rng.binomial(N_users - x, login_p)
    return stays + logins

def simulate_state_dep(x0, N, rng):
    states = np.zeros(N + 1, dtype=int)
    states[0] = x0
    x = x0
    for t in range(1, N + 1):
        x = step_state_dep(x, rng)
        states[t] = x
    return states

N_steps = 10000
x0 = 0

states_sd = simulate_state_dep(x0, N_steps, rng)
counts_sd = np.bincount(states_sd, minlength=N_users + 1)
dist_sd = counts_sd / counts_sd.sum()

print("Suma rozkładu (state-dependent):", dist_sd.sum())

plt.figure(figsize=(10,6))
plt.plot(dist_const, label="Stałe P(logout)=0.5 (Zad. C)")
plt.plot(dist_sd, label="Logout zależy od x (Zad. D)")
plt.grid(True)
plt.xlabel("Liczba zalogowanych użytkowników")
plt.ylabel("Prawdopodobieństwo empiryczne")
plt.title("Porównanie rozkładów – Zadanie C vs D")
plt.legend()
plt.show()
