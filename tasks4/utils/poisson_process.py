from __future__ import annotations

import math
import numpy as np

from .exponential import sample_exponential_inverse_cdf


def simulate_poisson_path(
    lam: float,
    t_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Symuluje jedną trajektorię procesu Poissona N(t) do czasu t_max
    poprzez sumowanie czasów międzyzdarzeniowych (Exp(lam)).

    Zwraca:
    - times: czasy zdarzeń (włącznie z 0 i t_max dla łatwego rysowania schodków)
    - counts: N(t) na odcinkach (schodki)
    """
    if lam <= 0 or t_max <= 0:
        raise ValueError("lam i t_max muszą być > 0")

    t = 0.0
    event_times = []
    while True:
        dt = float(sample_exponential_inverse_cdf(lam, size=1, rng=rng)[0])
        t += dt
        if t > t_max:
            break
        event_times.append(t)

    # Do wykresu schodkowego: start 0, kolejne skoki, koniec t_max
    times = np.array([0.0] + event_times + [t_max], dtype=float)
    counts = np.array(list(range(0, len(event_times) + 1)) + [len(event_times)], dtype=int)
    return times, counts


def simulate_poisson_counts_at_times(
    n_traj: int,
    times: np.ndarray,
    lam: float,
    rng: np.random.Generator,
    t_max: float | None = None,
) -> np.ndarray:
    """
    Dla n_traj trajektorii procesu Poissona (generowanych przez inter-arrivals Exp(lam)),
    zwraca N(t) dla każdego t w 'times'.

    Wynik: tablica (n_traj, len(times)).
    """
    if n_traj <= 0:
        raise ValueError("n_traj musi być > 0")
    if lam <= 0:
        raise ValueError("lam musi być > 0")
    if times.ndim != 1:
        raise ValueError("times musi być wektorem 1D")

    times_sorted = np.array(times, dtype=float)
    if np.any(times_sorted < 0):
        raise ValueError("czasy muszą być >= 0")

    t_max_eff = float(np.max(times_sorted) if t_max is None else max(float(t_max), float(np.max(times_sorted))))
    out = np.zeros((n_traj, times_sorted.size), dtype=int)

    # Oczekiwana liczba zdarzeń ~ lam*t_max_eff, tu lam=1, t_max<=90 => ~90.
    # Bierzemy zapas, a gdy za mało, dogenerowujemy w pętli.
    mean_events = lam * t_max_eff
    base_m = int(math.ceil(mean_events + 10.0 * math.sqrt(max(mean_events, 1.0)) + 50.0))
    base_m = max(base_m, 64)

    for i in range(n_traj):
        m = base_m
        while True:
            dts = sample_exponential_inverse_cdf(rate=lam, size=m, rng=rng)
            ts = np.cumsum(dts)
            if ts[-1] >= t_max_eff:
                break
            # rzadko: dogeneruj, jeśli jeszcze nie przekroczyliśmy t_max
            m = int(m * 1.5) + 1

        # Liczba zdarzeń do każdego t: ile ts <= t
        out[i, :] = np.searchsorted(ts, times_sorted, side="right")

    return out


def poisson_pmf_truncated(mu: float, tail_prob: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """
    Teoretyczny PMF Poisson(mu) obcięty do takiego K_max, żeby ogon był mały.
    Liczy rekurencyjnie, bez SciPy.

    Zwraca:
    - k_vals: 0..K_max
    - pmf: P(N=k)
    """
    if mu < 0:
        raise ValueError("mu musi być >= 0")
    if mu == 0:
        return np.array([0], dtype=int), np.array([1.0], dtype=float)

    pmf0 = math.exp(-mu)
    pmfs = [pmf0]
    cdf = pmf0
    k = 0
    # rekurencja: p(k+1) = p(k) * mu/(k+1)
    while 1.0 - cdf > tail_prob:
        k += 1
        pmfs.append(pmfs[-1] * mu / k)
        cdf += pmfs[-1]

        # bezpiecznik
        if k > int(mu + 50 * math.sqrt(mu) + 5000):
            break

    k_vals = np.arange(0, len(pmfs), dtype=int)
    pmf = np.array(pmfs, dtype=float)
    # normalizacja (w razie obcięcia)
    pmf /= pmf.sum()
    return k_vals, pmf

