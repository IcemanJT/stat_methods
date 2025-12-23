"""
MS_Zestaw4_2025 — rozwiązanie w Python 3 (bez SciPy).

Uruchomienie:
    python main.py

Co program generuje:
- Zadanie A:
  * przykład trajektorii procesu Poissona (skokowy wykres N(t))
  * rozkłady empiryczne N(t) dla t = 1, 20, 90 (10^4 trajektorii) + porównanie z PMF Poissona
- Zadanie B:
  * dla 3 przypadków: wykres liczby zadań w kolejce od czasu, oraz liczby wykonanych zadań od czasu
- Zadanie C:
  * weryfikacja prawa Little'a dla 3 przypadków (wydruk E(R), E(x), E(R)*λA, błąd wzgl.)
- Zadanie D:
  * wykresy: E(x) od λA, E(x) od λS, E(x) od r=λA/λS (z uśrednianiem po wielu symulacjach)

Uwagi dot. stabilności:
- Dla M/M/1 stabilność (w stanie stacjonarnym) wymaga ρ = λA/λS < 1.
- Przypadek II ma ρ > 1, więc kolejka rośnie; estymaty są „dla skończonego horyzontu".
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from utils.rng import make_rng, spawn_rngs
from utils.poisson_process import (
    simulate_poisson_path,
    simulate_poisson_counts_at_times,
    poisson_pmf_truncated,
)
from utils.queue_simulation import simulate_mm1_path, simulate_mm1_metrics
from utils.metrics import (
    summarize_little_law,
    estimate_ex_over_grid_lamA,
    estimate_ex_over_grid_lamS,
    estimate_ex_over_grid_r,
)
from utils.plotting import (
    plot_poisson_trajectory,
    plot_empirical_vs_theoretical_pmf,
    plot_queue_and_completions,
    plot_ex_curves,
)


def task_a(rng: np.random.Generator) -> None:
    lam = 1.0  # 1/min
    t_max = 90.0
    sample_times = np.array([1.0, 20.0, 90.0], dtype=float)

    # Przykładowa trajektoria
    times, counts = simulate_poisson_path(lam=lam, t_max=t_max, rng=rng)
    fig1 = plot_poisson_trajectory(times, counts, lam=lam, t_max=t_max)
    fig1.show()

    # 10^4 trajektorii — rozkład N(t) w zadanych czasach
    n_traj = 10_000
    counts_at = simulate_poisson_counts_at_times(
        n_traj=n_traj,
        times=sample_times,
        lam=lam,
        rng=rng,
        t_max=t_max,
    )

    # Porównanie z rozkładem Poissona (PMF)
    for j, t in enumerate(sample_times):
        k_vals, pmf_theory = poisson_pmf_truncated(mu=lam * t, tail_prob=1e-6)
        fig = plot_empirical_vs_theoretical_pmf(
            samples=counts_at[:, j],
            k_vals=k_vals,
            pmf_theory=pmf_theory,
            title=f"Zadanie A: N(t) dla t={t:g} min (10^4 trajektorii), λ={lam:g} 1/min",
        )
        fig.show()


def task_b(rng: np.random.Generator) -> None:
    cases = [
        ("I", 1 / 20, 1 / 15),
        ("II", 1 / 20, 1 / 100),
        ("III", 1 / 20, 1 / 5),
    ]
    T = 500.0  # horyzont do wykresów (czytelny)

    for label, lamA, lamS in cases:
        path = simulate_mm1_path(lamA=lamA, lamS=lamS, T=T, rng=rng)
        fig = plot_queue_and_completions(
            path,
            title_prefix=f"Zadanie B ({label})",
            lamA=lamA,
            lamS=lamS,
        )
        fig.show()


def task_c(rng: np.random.Generator) -> None:
    cases = [
        ("I", 1 / 20, 1 / 15),
        ("II", 1 / 20, 1 / 100),
        ("III", 1 / 20, 1 / 5),
    ]

    n_rep = 1000
    T = 10_000.0
    warmup = 1000.0  # redukcja wpływu rozruchu na średnią czasową

    rngs = spawn_rngs(rng, n_rep)

    print("\nZadanie C: weryfikacja prawa Little'a (E(R)*λA ≈ E(x))")
    print(f"  Liczba symulacji: {n_rep}, horyzont: T={T:g}, warmup={warmup:g}\n")

    for label, lamA, lamS in cases:
        ers = []
        exs = []
        for rr in rngs:
            m = simulate_mm1_metrics(lamA=lamA, lamS=lamS, T=T, warmup=warmup, rng=rr)
            ers.append(m["E_R"])
            exs.append(m["E_x"])

        summary = summarize_little_law(
            E_R=float(np.mean(ers)),
            E_x=float(np.mean(exs)),
            lamA=lamA,
        )

        rho = lamA / lamS
        print(f"Przypadek {label}: λA={lamA:.6f}, λS={lamS:.6f}, ρ={rho:.3f}")
        print(f"  E(R)      ≈ {summary['E_R']:.6f}")
        print(f"  E(x)      ≈ {summary['E_x']:.6f}")
        print(f"  E(R)*λA   ≈ {summary['E_R_lamA']:.6f}")
        print(f"  błąd abs  ≈ {summary['abs_err']:.6f}")
        print(f"  błąd wzgl ≈ {summary['rel_err']:.6%}\n")


def task_d(rng: np.random.Generator) -> None:
    # Uzasadnienie zakresów (w komentarzu):
    # - Dla M/M/1 sensowna jest okolica ρ<1 (stacjonarność).
    # - Pokazujemy też zbliżanie się do ρ≈1, gdzie E(x) rośnie.
    # - Dla stabilności wyników: uśrednianie po wielu powtórzeniach + warmup.
    T = 8000.0
    warmup = 1000.0
    n_rep = 300  # kompromis stabilność/czas

    lamA_fixed = 1 / 20  # 0.05

    # (a) E(x) od λA przy stałym λS
    lamS_for_lamA = 1 / 15  # ~0.0667 => ρ rośnie wraz z λA
    lamA_grid = np.linspace(0.01, 0.09, 17)  # obejmuje okolice ρ≈1
    ex_a = estimate_ex_over_grid_lamA(
        lamA_grid=lamA_grid,
        lamS=lamS_for_lamA,
        T=T,
        warmup=warmup,
        n_rep=n_rep,
        rng=rng,
    )

    # (b) E(x) od λS przy stałym λA
    lamS_grid = np.linspace(0.02, 0.20, 19)
    ex_b = estimate_ex_over_grid_lamS(
        lamS_grid=lamS_grid,
        lamA=lamA_fixed,
        T=T,
        warmup=warmup,
        n_rep=n_rep,
        rng=rng,
    )

    # (c) E(x) od r=λA/λS przy stałym λA (zmieniamy λS = λA/r)
    r_grid = np.linspace(0.2, 1.2, 21)  # przejście przez ρ=1
    ex_c, lamS_from_r = estimate_ex_over_grid_r(
        r_grid=r_grid,
        lamA=lamA_fixed,
        T=T,
        warmup=warmup,
        n_rep=n_rep,
        rng=rng,
    )

    fig = plot_ex_curves(
        lamA_grid=lamA_grid,
        ex_a=ex_a,
        lamS_for_lamA=lamS_for_lamA,
        lamS_grid=lamS_grid,
        ex_b=ex_b,
        lamA_fixed=lamA_fixed,
        r_grid=r_grid,
        ex_c=ex_c,
        lamS_from_r=lamS_from_r,
        n_rep=n_rep,
        T=T,
        warmup=warmup,
    )
    fig.show()


def main() -> None:
    rng = make_rng(seed=12345)

    task_a(rng)
    task_b(rng)
    task_c(rng)
    task_d(rng)

    # Jeśli uruchamiasz w trybie skryptu, to to pomoże utrzymać okna wykresów.
    plt.show()


if __name__ == "__main__":
    main()

