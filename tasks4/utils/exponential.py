from __future__ import annotations

import numpy as np


def sample_exponential_inverse_cdf(
    rate: float,
    size: int | tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generowanie Exp(rate) metodą odwracania dystrybuanty:
        T = -ln(1-U)/rate, U~Uniform(0,1)

    rate > 0
    """
    if rate <= 0:
        raise ValueError("rate musi być > 0")

    u = rng.random(size=size)
    # u in [0,1), więc 1-u w (0,1] i log jest ok
    return -np.log1p(-u) / rate

