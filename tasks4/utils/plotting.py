from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_poisson_trajectory(times: np.ndarray, counts: np.ndarray, lam: float, t_max: float) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.step(times, counts, where="post", linewidth=2)
    ax.set_title(f"Zadanie A: przykładowa trajektoria procesu Poissona (λ={lam:g} 1/min), t_max={t_max:g}")
    ax.set_xlabel("t [min]")
    ax.set_ylabel("N(t)")
    ax.set_xlim(0, t_max)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_empirical_vs_theoretical_pmf(
    samples: np.ndarray,
    k_vals: np.ndarray,
    pmf_theory: np.ndarray,
    title: str,
) -> plt.Figure:
    # empiryczne P(N=k)
    max_k = int(max(np.max(samples), np.max(k_vals)))
    counts = np.bincount(samples.astype(int), minlength=max_k + 1)
    pmf_emp = counts / counts.sum()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Empiryczne — słupki
    ks_emp = np.arange(0, pmf_emp.size, dtype=int)
    ax.bar(ks_emp, pmf_emp, alpha=0.6, label="Empiryczny (symulacja)")

    # Teoretyczne — linia z markerami
    ax.plot(k_vals, pmf_theory, marker="o", linestyle="-", label="Teoretyczny Poisson(λt)")

    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel("P(N(t)=k)")
    ax.set_xlim(0, max(max_k, int(k_vals[-1])) + 1)
    ax.set_ylim(0, max(np.max(pmf_emp), np.max(pmf_theory)) * 1.15 + 1e-12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_queue_and_completions(
    path: dict[str, np.ndarray],
    title_prefix: str,
    lamA: float,
    lamS: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    t_queue = path["t_queue"]
    q_queue = path["q_queue"]
    t_done = path["t_done"]
    done = path["done"]

    ax1.step(t_queue, q_queue, where="post", linewidth=2, label="Zadania w kolejce")
    ax1.set_title(f"{title_prefix}: liczba zadań w kolejce od czasu (λA={lamA:.6f}, λS={lamS:.6f})")
    ax1.set_xlabel("t")
    ax1.set_ylabel("q(t) (kolejka)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.step(t_done, done, where="post", linewidth=2, label="Wykonane zadania")
    ax2.set_title(f"{title_prefix}: liczba wykonanych zadań od czasu (λA={lamA:.6f}, λS={lamS:.6f})")
    ax2.set_xlabel("t")
    ax2.set_ylabel("done(t)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    return fig


def plot_ex_curves(
    lamA_grid: np.ndarray,
    ex_a: np.ndarray,
    lamS_for_lamA: float,
    lamS_grid: np.ndarray,
    ex_b: np.ndarray,
    lamA_fixed: float,
    r_grid: np.ndarray,
    ex_c: np.ndarray,
    lamS_from_r: np.ndarray,
    n_rep: int,
    T: float,
    warmup: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(lamA_grid, ex_a, marker="o", linestyle="-", label=f"λS stałe = {lamS_for_lamA:.6f}")
    ax1.set_title(f"Zadanie D(a): E(liczba zadań w systemie) od λA (uśrednianie n={n_rep}, T={T:g}, warmup={warmup:g})")
    ax1.set_xlabel("λA")
    ax1.set_ylabel("E(x)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(lamS_grid, ex_b, marker="o", linestyle="-", label=f"λA stałe = {lamA_fixed:.6f}")
    ax2.set_title(f"Zadanie D(b): E(liczba zadań w systemie) od λS (uśrednianie n={n_rep})")
    ax2.set_xlabel("λS")
    ax2.set_ylabel("E(x)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(r_grid, ex_c, marker="o", linestyle="-", label="λS = λA/r")
    ax3.axvline(1.0, linestyle="--", linewidth=1, label="r=1 (granica stabilności)")
    ax3.set_title(f"Zadanie D(c): E(liczba zadań w systemie) od r=λA/λS (λA={lamA_fixed:.6f})")
    ax3.set_xlabel("r")
    ax3.set_ylabel("E(x)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig.tight_layout()
    return fig

