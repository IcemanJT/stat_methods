"""
zadanie_d.py

Zadanie D:
- Hybrydowa metoda eliminacji z funkcją Lorentza (Cauchy'ego) jako obwiednią dla N(0,1).
- Funkcja gęstości Lorentza: g(x) = 1/pi * 1/(1+x^2), obwiednia C * g(x).
- Krok 1: 10^4 kandydatów, zliczenie odsetka akceptacji, pomiar czasu.
- Krok 2: generowanie aż do uzyskania 10^4 zaakceptowanych próbek,
  zliczenie odsetka akceptacji i pomiar czasu.
- Porównanie efektywności (czas + odsetek akceptacji) z metodą z zadania C.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from utils.pdfs import normal_pdf, lorentz_pdf, lorentz_envelope_pdf
from utils.simulations import (
    rejection_sample_normal_lorentz_fixed_candidates,
    rejection_sample_normal_lorentz_until_n,
)
from utils.plotting import plot_hist_with_pdf


def run_task_d():
    rng = np.random.default_rng(123)

    n_candidates_step1 = 10_000
    n_accepted_target_step2 = 10_000
    C = 1.6  # stała obwiedni, >= sup_x f(x)/g(x)

    print("Zadanie D – hybrydowa metoda eliminacji z obwiednią Lorentza dla N(0,1)")

    # --- KROK 1: 10^4 kandydatów ---
    t0 = time.time()
    accepted_step1, n_accept1 = rejection_sample_normal_lorentz_fixed_candidates(
        n_candidates_step1,
        C=C,
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

    # Opcjonalny histogram zaakceptowanych z Kroku 1
    if n_accept1 > 0:
        plot_hist_with_pdf(
            accepted_step1,
            pdf_func=normal_pdf,
            x_range=(-4, 4),
            bins=80,
            title=f"Zadanie D – Krok 1: zaakceptowane próbki (obwiednia Lorentza, C={C})",
            xlabel="x",
            ylabel="Gęstość",
        )

    # --- KROK 2: generujemy aż do uzyskania 10^4 zaakceptowanych próbek ---
    t0 = time.time()
    accepted_step2, total_candidates_step2 = rejection_sample_normal_lorentz_until_n(
        n_accepted_target_step2,
        C=C,
        batch_size=100,
        rng=rng,
    )
    t1 = time.time()

    acceptance_rate2 = accepted_step2.size / total_candidates_step2
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
        title=f"Zadanie D – Krok 2: 10^4 zaakceptowanych próbek (obwiednia Lorentza, C={C})",
        xlabel="x",
        ylabel="Gęstość",
    )

    plt.show()

    print("\nKomentarz:")
    print("- Obwiednia Lorentza jest lepiej dopasowana do ogona rozkładu normalnego niż prostokąt,")
    print("  dzięki czemu znacznie większy procent kandydatów jest akceptowany.")
    print("- Oczekujemy wyższego odsetka akceptacji oraz krótszego czasu generacji 10^4 próbek")
    print("  w porównaniu z metodą prostokątną z zadania C.")

if __name__ == "__main__":
    run_task_d()
