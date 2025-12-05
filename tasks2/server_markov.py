import numpy as np

# ===========================================
# LOGIKA DLA 2-UŻYTKOWNIKOWEGO MODELU (stany 0,1,2)
# ===========================================

def build_transition_matrix_two_users(p_login=0.2, p_stay_logged_in=0.5):
    """
    Buduje macierz przejścia P dla 2 niezależnych użytkowników.
    p_login: P(logowania) dla niezalogowanego
    p_stay_logged_in: P(pozostania zalogowanym) dla zalogowanego
    Zwraca macierz 3x3 dla stanów {0,1,2}.
    """
    p_logout = 1.0 - p_stay_logged_in

    # Stan 0: obaj niezalogowani
    # X ~ Bin(2, p_login)
    P00 = (1 - p_login) ** 2
    P01 = 2 * p_login * (1 - p_login)
    P02 = p_login ** 2

    # Stan 2: obaj zalogowani
    # Y ~ Bin(2, p_stay_logged_in)
    P20 = (1 - p_stay_logged_in) ** 2
    P21 = 2 * p_stay_logged_in * (1 - p_stay_logged_in)
    P22 = p_stay_logged_in ** 2

    # Stan 1: jeden zalogowany, jeden nie
    # Rozpisujemy 4 scenariusze:
    #  (stay_in, stay_out), (stay_in, login), (logout, stay_out), (logout, login)
    # i liczymy do stanów 0,1,2.
    P10 = p_logout * (1 - p_login)                  # (logout, stay_out)
    P12 = p_stay_logged_in * p_login                # (stay_in, login)
    P11 = 1.0 - P10 - P12                           # reszta

    P = np.array([
        [P00, P01, P02],
        [P10, P11, P12],
        [P20, P21, P22]
    ])

    return P


def power_iteration(P, N_max=500, eps=1e-6):
    """
    Liczy kolejne potęgi P^N aż do N_max i sprawdza konwergencję
    względem normy max: ||P^N - P^{N-1}||_∞ < eps.
    Zwraca:
      P_powers  – lista macierzy [P^0, P^1, ..., P^N_max]
      P_lim     – przybliżona macierz graniczna (ostatnia)
      converged_N – pierwsze N spełniające kryterium (lub None)
    """
    num_states = P.shape[0]
    P_prev = np.eye(num_states)
    P_powers = [P_prev]
    converged_N = None

    for N in range(1, N_max + 1):
        P_curr = P_prev @ P
        P_powers.append(P_curr)

        diff = np.max(np.abs(P_curr - P_prev))
        if diff < eps and converged_N is None:
            converged_N = N

        P_prev = P_curr

    P_lim = P_powers[-1]
    return P_powers, P_lim, converged_N


def simulate_markov_chain(P, x0, N_steps, rng=None):
    """
    Symuluje łańcuch Markowa o macierzy przejścia P.
    x0      – stan początkowy (indeks 0..n-1)
    N_steps – liczba kroków
    rng     – instancja Generator numpy (opcjonalne)
    Zwraca tablicę odwiedzonych stanów (długości N_steps+1).
    """
    if rng is None:
        rng = np.random.default_rng()

    num_states = P.shape[0]
    states = np.empty(N_steps + 1, dtype=int)
    states[0] = x0
    cur = x0

    for t in range(1, N_steps + 1):
        cur = rng.choice(num_states, p=P[cur])
        states[t] = cur

    return states


def empirical_probabilities_from_path(states, num_states):
    """
    Dla zadania B: z wektora odwiedzonych stanów liczy empiryczne
    prawdopodobieństwa w kolejnych krokach (0..N).
    Zwraca macierz [N+1, num_states], gdzie wiersz t to
    wektor częstości po t krokach.
    """
    N_steps = len(states) - 1
    counts = np.zeros((N_steps + 1, num_states), dtype=float)
    running = np.zeros(num_states, dtype=float)

    for t in range(N_steps + 1):
        running[states[t]] += 1
        if t > 0:
            counts[t] = running / t

    return counts

# ===========================================
# LOGIKA DLA MODELU 100 UŻYTKOWNIKÓW
# ===========================================

def step_100_users_const(x, rng, N_users=100, p_login=0.2, p_stay_logged_in=0.5):
    """
    Jednokrokowa ewolucja stanu x (liczba zalogowanych) przy
    stałych prawdopodobieństwach:
      - p_login: P(logowania) niezalogowanego
      - p_stay_logged_in: P(pozostania zalogowanym)
    """
    stays = rng.binomial(x, p_stay_logged_in)
    logins = rng.binomial(N_users - x, p_login)
    return stays + logins


def step_100_users_state_dependent(x, rng, N_users=100, p_login=0.2):
    """
    Jednokrokowa ewolucja stanu x przy P(stay | x) = 0.008*x + 0.1.
    """
    p_stay = 0.008 * x + 0.1
    p_stay = np.clip(p_stay, 0.0, 1.0)
    stays = rng.binomial(x, p_stay)
    logins = rng.binomial(N_users - x, p_login)
    return stays + logins


def simulate_100_users(step_func, x0, N_steps, rng=None, **step_kwargs):
    """
    Ogólna symulacja dla 100-użytkowników (lub N_users w step_kwargs)
    step_func  – np. step_100_users_const albo step_100_users_state_dependent
    x0         – stan początkowy
    N_steps    – liczba kroków
    step_kwargs – dodatkowe argumenty przekazywane do step_func
    """
    if rng is None:
        rng = np.random.default_rng()

    states = np.empty(N_steps + 1, dtype=int)
    states[0] = x0
    x = x0

    for t in range(1, N_steps + 1):
        x = step_func(x, rng, **step_kwargs)
        states[t] = x

    return states


def empirical_distribution_from_states(states, N_users):
    """
    Zlicza częstości odwiedzin każdego stanu 0..N_users.
    Zwraca wektor rozkładu empirycznego długości N_users+1.
    """
    counts = np.bincount(states, minlength=N_users + 1)
    dist = counts / counts.sum()
    return dist
