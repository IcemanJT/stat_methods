from __future__ import annotations

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Tworzy Generator NumPy z ustawialnym ziarnem dla powtarzalności."""
    return np.random.default_rng(seed)


def spawn_rngs(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """
    Tworzy n niezależnych generatorów (pod-strumieni) z jednego RNG.
    Pomaga w powtarzalnych eksperymentach Monte Carlo.
    """
    seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64)
    return [np.random.default_rng(int(s)) for s in seeds]

