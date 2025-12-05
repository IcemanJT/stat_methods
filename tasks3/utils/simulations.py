"""
simulations.py

Funkcje do symulacji metodą eliminacji/akceptacji (rejection sampling):
- klasyczna metoda z prostokątną obwiednią dla N(0,1)
- hybrydowa metoda z obwiednią Lorentza (Cauchy'ego)
"""

import numpy as np

from .generators import generate_cauchy
from .pdfs import normal_pdf, lorentz_pdf


def rejection_sample_normal_rectangle_fixed_candidates(
    n_candidates: int,
    x_min: float = -5.0,
    x_max: float = 5.0,
    rng: np.random.Generator | None = None
) -> tuple[np.ndarray, int]:
    """
    Klasyczna metoda hit-and-miss (eliminacji) dla N(0,1) na prostokącie:
    (x, y) ~ U([x_min, x_max] x [0, f_max]), gdzie f_max = f(0).
    Zwraca:
    - zaakceptowane wartości x,
    - liczbę zaakceptowanych punktów.
    """
    if rng is None:
        rng = np.random.default_rng()

    f_max = normal_pdf(0.0)  # maksimum gęstości N(0,1)

    xs = rng.uniform(x_min, x_max, size=n_candidates)
    ys = rng.uniform(0.0, f_max, size=n_candidates)

    fx = normal_pdf(xs)
    accepted_mask = ys <= fx
    accepted_x = xs[accepted_mask]

    return accepted_x, accepted_x.size


def rejection_sample_normal_rectangle_until_n(
    n_accept: int,
    x_min: float = -5.0,
    x_max: float = 5.0,
    batch_size: int = 10_000,
    rng: np.random.Generator | None = None
) -> tuple[np.ndarray, int]:
    """
    Klasyczna metoda hit-and-miss dla N(0,1) na prostokącie:
    generuje kandydatów aż do uzyskania n_accept zaakceptowanych punktów.
    Zwraca:
    - zaakceptowane wartości x (dokładnie n_accept),
    - całkowitą liczbę wygenerowanych kandydatów.
    """
    if rng is None:
        rng = np.random.default_rng()

    f_max = normal_pdf(0.0)

    accepted = []
    total_candidates = 0

    while len(accepted) < n_accept:
        xs = rng.uniform(x_min, x_max, size=batch_size)
        ys = rng.uniform(0.0, f_max, size=batch_size)
        fx = normal_pdf(xs)
        accepted_mask = ys <= fx
        acc_x = xs[accepted_mask]
        accepted.append(acc_x)
        total_candidates += batch_size

    accepted_all = np.concatenate(accepted)
    return accepted_all[:n_accept], total_candidates


def rejection_sample_normal_lorentz_fixed_candidates(
    n_candidates: int,
    C: float = 1.6,
    rng: np.random.Generator | None = None
) -> tuple[np.ndarray, int]:
    """
    Hybrydowa metoda eliminacji: proposal g(x) = Lorentz(0,1), envelope C*g(x).

    Krok:
    - generujemy X ~ Cauchy(0,1)
    - generujemy U ~ U(0,1)
    - akceptujemy X, jeśli U <= f(X)/(C * g(X))

    Tu generujemy n_candidates kandydatów (X,U) i zwracamy:
    - akceptowane wartości,
    - liczbę akceptacji.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = generate_cauchy(n_candidates, rng=rng)
    u = rng.random(size=n_candidates)

    f_x = normal_pdf(x)
    g_x = lorentz_pdf(x)
    accept_prob = f_x / (C * g_x)
    accept_prob = np.clip(accept_prob, 0.0, 1.0)

    accepted_mask = u <= accept_prob
    accepted_x = x[accepted_mask]

    return accepted_x, accepted_x.size


def rejection_sample_normal_lorentz_until_n(
    n_accept: int,
    C: float = 1.6,
    batch_size: int = 10_000,
    rng: np.random.Generator | None = None
) -> tuple[np.ndarray, int]:
    """
    Hybrydowa metoda eliminacji z obwiednią Lorentza:
    generuje kandydatów aż do uzyskania n_accept zaakceptowanych punktów.
    Zwraca:
    - zaakceptowane wartości x (dokładnie n_accept),
    - całkowitą liczbę wygenerowanych kandydatów.
    """
    if rng is None:
        rng = np.random.default_rng()

    accepted = []
    total_candidates = 0

    while len(accepted) < n_accept:
        x = generate_cauchy(batch_size, rng=rng)
        u = rng.random(size=batch_size)

        f_x = normal_pdf(x)
        g_x = lorentz_pdf(x)
        accept_prob = f_x / (C * g_x)
        accept_prob = np.clip(accept_prob, 0.0, 1.0)

        accepted_mask = u <= accept_prob
        acc_x = x[accepted_mask]
        accepted.append(acc_x)
        total_candidates += batch_size

    accepted_all = np.concatenate(accepted)
    return accepted_all[:n_accept], total_candidates
