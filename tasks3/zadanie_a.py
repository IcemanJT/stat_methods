"""
zadanie_a.py

Zadanie A:
- Zaimplementowanie generatora N(0,1) metodą Boxa–Mullera (w generators.py).
- Wygenerowanie dużej liczby próbek z N(0,1).
- Narysowanie histogramu oraz nałożenie gęstości teoretycznej N(0,1).
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.generators import generate_normal_box_muller
from utils.pdfs import normal_pdf
from utils.plotting import plot_hist_with_pdf


def run_task_a():
    rng = np.random.default_rng(123)

    # Liczba próbek
    n_samples = 100_000

    # Generacja z N(0,1) metodą Boxa–Mullera
    samples = generate_normal_box_muller(n_samples, rng=rng)

    # Prosta weryfikacja numeryczna: średnia i wariancja
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)

    print("Zadanie A:")
    print(f"Liczba próbek: {n_samples}")
    print(f"Średnia z próbek ~ {sample_mean:.4f} (teoretycznie 0)")
    print(f"Wariancja z próbek ~ {sample_var:.4f} (teoretycznie 1)")

    # Histogram + gęstość teoretyczna
    plot_hist_with_pdf(
        samples,
        pdf_func=normal_pdf,
        x_range=(-4, 4),
        bins=80,
        title="Zadanie A – N(0,1) metodą Boxa–Mullera",
        xlabel="x",
        ylabel="Gęstość",
    )

    plt.show()


if __name__ == "__main__":
    run_task_a()
