"""
generators.py

Generatory liczb pseudolosowych:
- jednostajny U(a,b)
- wykładniczy Exp(lam)
- normalny N(0,1) metodą Boxa–Mullera
- Cauchy'ego (Lorentza)
"""

import numpy as np


def _get_rng(rng: np.random.Generator | None = None) -> np.random.Generator:
    """Zwraca przekazany generator lub tworzy domyślny."""
    if rng is None:
        rng = np.random.default_rng()
    return rng


def generate_uniform(n: int, low: float = 0.0, high: float = 1.0,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Generator liczb z rozkładu jednostajnego U(low, high)."""
    rng = _get_rng(rng)
    return rng.random(size=n) * (high - low) + low


def generate_exponential(n: int, lam: float = 1.0,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """Generator liczb z rozkładu wykładniczego Exp(lam) metodą odwrotnej dystrybuanty."""
    rng = _get_rng(rng)
    u = rng.random(size=n)
    # U ~ U(0,1) => X = - (1/lam) * ln(1-U) ~ Exp(lam)
    return -np.log1p(-u) / lam


def generate_normal_box_muller(n: int,
                               rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generator liczb z N(0,1) metodą Boxa–Mullera.
    Zwraca dokładnie n próbek.
    """
    rng = _get_rng(rng)

    m = (n + 1) // 2  # liczba par
    u1 = rng.random(size=m)
    u2 = rng.random(size=m)

    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2

    z1 = r * np.cos(theta)
    z2 = r * np.sin(theta)

    z = np.empty(2 * m)
    z[0::2] = z1
    z[1::2] = z2

    return z[:n]


def generate_cauchy(n: int,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generator liczb z rozkładu Cauchy'ego(0,1) (Lorentza) metodą odwrotnej dystrybuanty.

    Jeśli U ~ U(0,1), to X = tan(pi(U - 1/2)) ~ Cauchy(0,1).
    """
    rng = _get_rng(rng)
    u = rng.random(size=n)
    return np.tan(np.pi * (u - 0.5))
