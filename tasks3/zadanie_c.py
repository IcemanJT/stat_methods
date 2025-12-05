"""
zadanie_c.py

Zadanie C:
- Zastosowanie klasycznej metody eliminacji/akceptacji (hit-and-miss) do generacji N(0,1)
  z użyciem prostokątnej obwiedni nad gęstością normalną.
- Krok 1: 10^4 kandydatów, zliczenie odsetka akceptacji, pomiar czasu.
- Krok 2: generowanie aż do uzyskania 10^4 zaakceptowanych próbek,
  zliczenie odsetka akceptacji i pomiar czasu.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from utils.pdfs import normal_pdf
from utils.simulations import (
    rejection_sample_normal_rectangle_fixed_candidates,
    rejection_sample_normal_rectangle_until_n,
)
from utils.plotting import plot_hist_with_pdf


def run_task_c():
    rng = np.random.default_rng(123)

    n_candidates_step1 = 10_000
    n_accepted_target_step2 = 10_000

    print("Zadanie C – klasyczna metoda eliminacji z prostokątną obwiednią nad N(0,1)")

    # --- KROK 1: 10^4 kandydatów ---
    t0 = time.time()
    accepted_step1, n_accept1 = rejection_sample_normal_rectangle_fixed_candidates(
        n_candidates_step1,
        x_min=-5.0,
        x_max=5.0,
        rng=rng,
    )
    t1 = time.time()

    acceptance_rate1 = n_accept1 / n_candidates_step1
    elapsed1 = t1 - t0

    print(f"\nKrok 1:")
    print(f"  Kandydatów: {n_candidates_step1}")
    print(f"  Zaakceptowanych: {n_accept1}")
    print(f"  Odsetek akceptacji: {acceptance_rate1 * 100:.2f}%")
    print(f"  Czas wykonania: {elapsed1:.4f} s")

    # Opcjonalnie: histogram zaakceptowanych z Kroku 1
    if n_accept1 > 0:
        plot_hist_with_pdf(
            accepted_step1,
            pdf_func=normal_pdf,
            x_range=(-4, 4),
            bins=80,
            title="Zadanie C – Krok 1: zaakceptowane próbki (prostokątna obwiednia)",
            xlabel="x",
            ylabel="Gęstość",
        )

    # --- KROK 2: generujemy aż do uzyskania 10^4 zaakceptowanych próbek ---
    t0 = time.time()
    accepted_step2, total_candidates_step2 = rejection_sample_normal_rectangle_until_n(
        n_accepted_target_step2,
        x_min=-5.0,
        x_max=5.0,
        batch_size=10_000,
        rng=rng,
    )
    t1 = time.time()

    acceptance_rate2 = n_accepted_target_step2 / total_candidates_step2
    elapsed2 = t1 - t0

    print(f"\nKrok 2:")
    print(f"  Docelowa liczba zaakceptowanych próbek: {n_accepted_target_step2}")
    print(f"  Faktycznie wygenerowanych kandydatów: {total_candidates_step2}")
    print(f"  Odsetek akceptacji: {acceptance_rate2 * 100:.2f}%")
    print(f"  Czas wykonania: {elapsed2:.4f} s")

    # Histogram zaakceptowanych próbek z Kroku 2
    plot_hist_with_pdf(
        accepted_step2,
        pdf_func=normal_pdf,
        x_range=(-4, 4),
        bins=80,
        title="Zadanie C – Krok 2: 10^4 zaakceptowanych próbek (prostokątna obwiednia)",
        xlabel="x",
        ylabel="Gęstość",
    )

    plt.show()

    print("\nKomentarz:")
    print("- Odsetek akceptacji jest dość niski, ponieważ prostokątna obwiednia ma spory 'pusty' obszar,")
    print("  w którym y > f(x) i wiele punktów zostaje odrzuconych.")
    print("- Czas generacji 10^4 zaakceptowanych próbek jest przez to stosunkowo długi.")
    print("- W zadaniu D porównamy tę metodę z odmianą wykorzystującą obwiednię Lorentza (Cauchy'ego).")


if __name__ == "__main__":
    run_task_c()
