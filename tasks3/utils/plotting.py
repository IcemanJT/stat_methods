"""
plotting.py

Funkcje do rysowania:
- histogramów
- histogram + nałożona gęstość (pdf)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_hist(samples: np.ndarray,
              bins: int = 50,
              density: bool = True,
              title: str | None = None,
              xlabel: str | None = None,
              ylabel: str = "Gęstość",
              logy: bool = False) -> None:
    """Rysuje prosty histogram 1D."""
    plt.figure()
    plt.hist(samples, bins=bins, density=density, edgecolor="black", alpha=0.7)
    if logy:
        plt.yscale("log")
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)


def plot_hist_with_pdf(samples: np.ndarray,
                       pdf_func,
                       x_range: tuple[float, float] = (-4.0, 4.0),
                       bins: int = 50,
                       title: str | None = None,
                       xlabel: str | None = None,
                       ylabel: str = "Gęstość",
                       n_points: int = 1000) -> None:
    """
    Rysuje histogram próbek oraz nakłada na niego funkcję gęstości pdf_func.
    pdf_func: funkcja pdf(x) -> gęstość.
    """
    plt.figure()
    plt.hist(samples, bins=bins, density=True, edgecolor="black",
             alpha=0.7, label="Histogram próbek")

    xs = np.linspace(x_range[0], x_range[1], n_points)
    ys = pdf_func(xs)
    plt.plot(xs, ys, linewidth=2.0, label="Gęstość teoretyczna")

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
