"""
pdfs.py

Funkcje gęstości:
- normalna N(mu, sigma^2)
- Lorentza (Cauchy'ego)
- obwiednia Lorentza C * g(x)
"""

import numpy as np


def normal_pdf(x: np.ndarray | float,
               mu: float = 0.0,
               sigma: float = 1.0) -> np.ndarray:
    """Gęstość rozkładu normalnego N(mu, sigma^2)."""
    x = np.asarray(x)
    coeff = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    z = (x - mu) / sigma
    return coeff * np.exp(-0.5 * z * z)


def lorentz_pdf(x: np.ndarray | float) -> np.ndarray:
    """
    Gęstość standardowego rozkładu Lorentza / Cauchy'ego(0,1):
        g(x) = 1 / (pi * (1 + x^2))
    """
    x = np.asarray(x)
    return 1.0 / (np.pi * (1.0 + x * x))


def lorentz_envelope_pdf(x: np.ndarray | float,
                         C: float = 1.6) -> np.ndarray:
    """
    Funkcja obwiedni: C * g(x), gdzie g(x) to gęstość Lorentza.
    Stała C powinna być >= sup_x f(x)/g(x) dla f=N(0,1).
    Numerycznie sup_x f/g ~ 1.52, więc C=1.6 jest bezpieczne.
    """
    return C * lorentz_pdf(x)
