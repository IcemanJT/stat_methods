"""
zadanie_b.py

Zadanie B składa się z dwóch części:

Część 1:
- Generowanie sum 10 i 100 liczb z rozkładów:
  a) normalnego (własny generator z zadania A),
  b) wykładniczego,
  c) jednostajnego na [0,1].
- Dla każdej kombinacji: histogram rozkładu sum.
- Komentarz: wyniki ilustrują Centralne Twierdzenie Graniczne (CLT).

Część 2:
- Wylosowanie dużej liczby liczb z rozkładów jak wyżej.
- Posortowanie próbek.
- Policzenie różnic między kolejnymi próbkami.
- Histogram różnic oraz ten sam histogram ze skalą logarytmiczną na osi Y.
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.generators import (
    generate_normal_box_muller,
    generate_exponential,
    generate_uniform,
)
from utils.plotting import plot_hist


def _part1_sums_clt(rng: np.random.Generator) -> None:
    n_sums = 50_000   # liczba sum
    sum_lengths = [10, 100]

    def gen_normal(shape):
        # shape może być int lub krotka -> obsłużymy generację macierzową
        size = shape
        return generate_normal_box_muller(np.prod(size), rng=rng).reshape(size)

    def gen_exponential(shape):
        return generate_exponential(np.prod(shape), lam=1.0, rng=rng).reshape(shape)

    def gen_uniform_(shape):
        return generate_uniform(np.prod(shape), low=0.0, high=1.0, rng=rng).reshape(shape)

    generators = {
        "normalny N(0,1)": gen_normal,
        "wykładniczy Exp(1)": gen_exponential,
        "jednostajny U(0,1)": gen_uniform_,
    }

    print("\nZadanie B – Część 1 (sumy):")
    print("Oczekujemy, że rozkład sum (po standaryzacji) zbliża się do normalnego (CLT).")

    for name, gen in generators.items():
        for m in sum_lengths:
            samples = gen((n_sums, m))
            sums = np.sum(samples, axis=1)

            title = f"Zadanie B – Część 1: sumy {m} próbek ({name})"
            plot_hist(
                sums,
                bins=80,
                density=True,
                title=title,
                xlabel=f"Suma {m} próbek",
                ylabel="Gęstość",
                logy=False,
            )

            print(f"  Rozkład sum {m} próbek ({name}):")
            print(f"    Średnia ~ {np.mean(sums):.4f}, wariancja ~ {np.var(sums):.4f}")


def _part2_ordered_differences(rng: np.random.Generator) -> None:
    n_samples = 100_000  # liczba próbek do sortowania

    def gen_normal(size):  # N(0,1)
        return generate_normal_box_muller(size, rng=rng)

    def gen_exponential_(size):  # Exp(1)
        return generate_exponential(size, lam=1.0, rng=rng)

    def gen_uniform_(size):  # U(0,1)
        return generate_uniform(size, low=0.0, high=1.0, rng=rng)

    generators = {
        "normalny N(0,1)": gen_normal,
        "wykładniczy Exp(1)": gen_exponential_,
        "jednostajny U(0,1)": gen_uniform_,
    }

    print("\nZadanie B – Część 2 (różnice między uporządkowanymi próbkami):")

    for name, gen in generators.items():
        samples = gen(n_samples)
        samples_sorted = np.sort(samples)
        diffs = np.diff(samples_sorted)

        # Histogram w skali liniowej
        title_lin = f"Zadanie B – Część 2: różnice między uporządkowanymi próbkami ({name})"
        plot_hist(
            diffs,
            bins=80,
            density=True,
            title=title_lin,
            xlabel="Różnice między kolejnymi próbkami",
            ylabel="Gęstość",
            logy=False,
        )

        # Histogram w skali logarytmicznej (oś Y)
        title_log = f"Zadanie B – Część 2: różnice (skala log Y) ({name})"
        plot_hist(
            diffs,
            bins=80,
            density=True,
            title=title_log,
            xlabel="Różnice między kolejnymi próbkami",
            ylabel="Gęstość (log)",
            logy=True,
        )

        print(f"  Rozkład różnic ({name}):")
        print(f"    Średnia różnica ~ {np.mean(diffs):.6e}")
        print(f"    Mediana różnicy ~ {np.median(diffs):.6e}")


def run_task_b():
    rng = np.random.default_rng(123)

    _part1_sums_clt(rng)
    _part2_ordered_differences(rng)

    plt.show()
    print("\nKomentarz (CLT):")
    print("Dla sum 10 i 100 niezależnych próbek z różnych rozkładów widać, że:")
    print("- rozkłady sum (po ewentualnej standaryzacji) przybliżają rozkład normalny,")
    print("- im więcej składników w sumie, tym lepsze przybliżenie – ilustracja Centralnego Twierdzenia Granicznego.")


if __name__ == "__main__":
    run_task_b()
