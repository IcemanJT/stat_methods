from __future__ import annotations

from collections import deque
import numpy as np

from .exponential import sample_exponential_inverse_cdf


def simulate_mm1_path(
    lamA: float,
    lamS: float,
    T: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Symulacja zdarzeniowa kolejki M/M/1 do czasu T.

    Zwraca dane do wykresów (schodki):
    - t_queue, q_queue : liczba zadań w kolejce (bez zadania w obsłudze)
    - t_done, done     : liczba wykonanych zadań od czasu
    """
    if lamA <= 0 or lamS <= 0 or T <= 0:
        raise ValueError("lamA, lamS, T muszą być > 0")

    t = 0.0
    in_system = 0  # liczba w systemie (kolejka + ewentualnie obsługa)
    done = 0

    t_next_arr = float(sample_exponential_inverse_cdf(lamA, 1, rng)[0])
    t_next_dep = np.inf  # brak zakończenia, dopóki nie ma obsługi

    t_queue = [0.0]
    q_queue = [0]  # kolejka bez serwera: max(in_system-1,0)

    t_done = [0.0]
    done_series = [0]

    while t < T:
        if t_next_arr <= t_next_dep and t_next_arr <= T:
            # ARRIVAL
            t = t_next_arr
            in_system += 1

            # zaplanuj kolejne przyjście
            t_next_arr = t + float(sample_exponential_inverse_cdf(lamA, 1, rng)[0])

            # jeśli serwer był pusty (in_system==1), startujemy obsługę
            if in_system == 1:
                t_next_dep = t + float(sample_exponential_inverse_cdf(lamS, 1, rng)[0])

            # aktualizacja kolejki (bez zadania w obsłudze)
            t_queue.append(t)
            q_queue.append(max(in_system - 1, 0))

        else:
            # DEPARTURE
            if t_next_dep > T:
                break
            t = t_next_dep
            done += 1
            in_system -= 1

            # zaplanuj następne zakończenie, jeśli ktoś czeka/obsługa trwa
            if in_system > 0:
                t_next_dep = t + float(sample_exponential_inverse_cdf(lamS, 1, rng)[0])
            else:
                t_next_dep = np.inf

            # do wykresów
            t_queue.append(t)
            q_queue.append(max(in_system - 1, 0))

            t_done.append(t)
            done_series.append(done)

    return {
        "t_queue": np.array(t_queue, dtype=float),
        "q_queue": np.array(q_queue, dtype=int),
        "t_done": np.array(t_done, dtype=float),
        "done": np.array(done_series, dtype=int),
    }


def simulate_mm1_metrics(
    lamA: float,
    lamS: float,
    T: float,
    warmup: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    Metryki do Little'a:
    - E_x: średnia czasowa liczby zadań w systemie po warmup (pole pod N(t)/czas)
    - E_R: średni czas w systemie dla zadań, które zakończyły się po warmup i przed T

    W przypadku niestabilnym (ρ>1) wyniki są „dla skończonego horyzontu".
    """
    if lamA <= 0 or lamS <= 0 or T <= 0:
        raise ValueError("lamA, lamS, T muszą być > 0")
    if not (0 <= warmup < T):
        raise ValueError("warmup musi spełniać 0 <= warmup < T")

    t = 0.0
    in_system = 0
    t_next_arr = float(sample_exponential_inverse_cdf(lamA, 1, rng)[0])
    t_next_dep = np.inf

    # kolejka czasów przyjścia dla zadań w systemie
    arrival_times = deque()

    # pole pod N(t) na odcinku [warmup, T]
    area = 0.0
    last_t = 0.0
    last_N = 0

    # sojourn times po warmup
    sojourn_sum = 0.0
    sojourn_cnt = 0

    def add_area_segment(t0: float, t1: float, N: int) -> None:
        nonlocal area
        a = max(t0, warmup)
        b = min(t1, T)
        if b > a:
            area += (b - a) * N

    while t < T:
        t_event = min(t_next_arr, t_next_dep, T)

        # dodaj pole za [last_t, t_event] z last_N
        add_area_segment(last_t, t_event, last_N)

        t = t_event
        last_t = t

        if t >= T:
            break

        if t_next_arr <= t_next_dep:
            # ARRIVAL
            in_system += 1
            arrival_times.append(t)

            t_next_arr = t + float(sample_exponential_inverse_cdf(lamA, 1, rng)[0])
            if in_system == 1:
                t_next_dep = t + float(sample_exponential_inverse_cdf(lamS, 1, rng)[0])

        else:
            # DEPARTURE
            in_system -= 1
            at = arrival_times.popleft()
            # uwzględniaj tylko zadania kończące się po warmup
            if t >= warmup:
                sojourn_sum += (t - at)
                sojourn_cnt += 1

            if in_system > 0:
                t_next_dep = t + float(sample_exponential_inverse_cdf(lamS, 1, rng)[0])
            else:
                t_next_dep = np.inf

        last_N = in_system

    denom = (T - warmup)
    E_x = area / denom if denom > 0 else float("nan")
    E_R = (sojourn_sum / sojourn_cnt) if sojourn_cnt > 0 else float("nan")

    return {"E_x": float(E_x), "E_R": float(E_R), "departures_used": float(sojourn_cnt)}

