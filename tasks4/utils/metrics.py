from __future__ import annotations

import numpy as np

from .rng import spawn_rngs
from .queue_simulation import simulate_mm1_metrics


def summarize_little_law(E_R: float, E_x: float, lamA: float) -> dict[str, float]:
    """
    Zwraca podsumowanie do wydruku: E(R), E(x), E(R)*λA, błąd abs i wzgl.
    """
    E_R_lamA = E_R * lamA
    abs_err = abs(E_R_lamA - E_x)
    rel_err = abs_err / abs(E_x) if E_x != 0 else float("inf")
    return {
        "E_R": float(E_R),
        "E_x": float(E_x),
        "E_R_lamA": float(E_R_lamA),
        "abs_err": float(abs_err),
        "rel_err": float(rel_err),
    }


def _estimate_ex_single(
    lamA: float,
    lamS: float,
    T: float,
    warmup: float,
    n_rep: int,
    rng: np.random.Generator,
) -> float:
    rngs = spawn_rngs(rng, n_rep)
    exs = []
    for rr in rngs:
        m = simulate_mm1_metrics(lamA=lamA, lamS=lamS, T=T, warmup=warmup, rng=rr)
        exs.append(m["E_x"])
    return float(np.mean(exs))


def estimate_ex_over_grid_lamA(
    lamA_grid: np.ndarray,
    lamS: float,
    T: float,
    warmup: float,
    n_rep: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.empty_like(lamA_grid, dtype=float)
    for i, lamA in enumerate(lamA_grid):
        out[i] = _estimate_ex_single(lamA=float(lamA), lamS=lamS, T=T, warmup=warmup, n_rep=n_rep, rng=rng)
    return out


def estimate_ex_over_grid_lamS(
    lamS_grid: np.ndarray,
    lamA: float,
    T: float,
    warmup: float,
    n_rep: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.empty_like(lamS_grid, dtype=float)
    for i, lamS in enumerate(lamS_grid):
        out[i] = _estimate_ex_single(lamA=lamA, lamS=float(lamS), T=T, warmup=warmup, n_rep=n_rep, rng=rng)
    return out


def estimate_ex_over_grid_r(
    r_grid: np.ndarray,
    lamA: float,
    T: float,
    warmup: float,
    n_rep: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dla zadanej siatki r = λA/λS i stałego λA, wylicza λS = λA/r i estymuje E(x).
    Zwraca (E(x) po r, odpowiadające λS).
    """
    lamS_vals = lamA / r_grid
    ex = np.empty_like(r_grid, dtype=float)
    for i, lamS in enumerate(lamS_vals):
        ex[i] = _estimate_ex_single(lamA=lamA, lamS=float(lamS), T=T, warmup=warmup, n_rep=n_rep, rng=rng)
    return ex, lamS_vals

