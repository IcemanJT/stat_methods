"""
Gambler's Ruin Problem - Base Module
Contains simulation functions and theoretical formulas
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


def simulate_game(a: int, b: int, pA: float, max_rounds: int = 100000) -> Tuple[bool, int, List[int]]:
    """
    Simulate a single Gambler's Ruin game.
    
    Parameters:
    -----------
    a : int
        Initial capital of player A
    b : int
        Initial capital of player B
    pA : float
        Probability that player A wins a single round
    max_rounds : int
        Maximum number of rounds to prevent infinite loops
    
    Returns:
    --------
    Tuple[bool, int, List[int]]
        (A_wins, num_rounds, capital_history)
        - A_wins: True if A wins, False if B wins
        - num_rounds: Number of rounds until game ends
        - capital_history: List of A's capital after each round
    """
    capital_A = a
    capital_B = b
    rounds = 0
    capital_history = [capital_A]
    
    while capital_A > 0 and capital_B > 0 and rounds < max_rounds:
        if np.random.random() < pA:
            capital_A += 1
            capital_B -= 1
        else:
            capital_A -= 1
            capital_B += 1
        
        rounds += 1
        capital_history.append(capital_A)
    
    A_wins = (capital_A > 0)
    return A_wins, rounds, capital_history


def simulate_multiple_games(a: int, b: int, pA: float, num_simulations: int = 10000) -> Dict:
    """
    Run multiple simulations and collect statistics.
    
    Returns:
    --------
    Dict with keys:
        - A_wins: List of booleans (True if A won)
        - num_rounds: List of round counts
        - P_ruin_A: Probability that A goes bankrupt
        - avg_rounds: Average number of rounds
    """
    A_wins_list = []
    num_rounds_list = []
    
    for _ in range(num_simulations):
        A_wins, rounds, _ = simulate_game(a, b, pA)
        A_wins_list.append(A_wins)
        num_rounds_list.append(rounds)
    
    P_ruin_A = 1 - np.mean(A_wins_list)
    avg_rounds = np.mean(num_rounds_list)
    
    return {
        'A_wins': A_wins_list,
        'num_rounds': num_rounds_list,
        'P_ruin_A': P_ruin_A,
        'avg_rounds': avg_rounds
    }


def theoretical_P_ruin_A(a: int, b: int, pA: float) -> float:
    """
    Calculate theoretical probability that player A goes bankrupt.
    
    Formula:
    - If pA != 0.5: P_ruin_A = ((q/p)^a - (q/p)^(a+b)) / (1 - (q/p)^(a+b))
    - If pA == 0.5: P_ruin_A = b / (a + b)
    
    where q = 1 - pA, p = pA
    """
    if pA == 0.5:
        return b / (a + b)
    else:
        q = 1 - pA
        p = pA
        ratio = q / p
        numerator = ratio**a - ratio**(a + b)
        denominator = 1 - ratio**(a + b)
        return numerator / denominator


def theoretical_expected_rounds(a: int, b: int, pA: float) -> float:
    """
    Calculate theoretical expected number of rounds.
    
    Formula:
    - If pA == 0.5: E[L] = a * b
    - If pA != 0.5: More complex formula involving q/p ratio
    """
    if pA == 0.5:
        return a * b
    else:
        q = 1 - pA
        p = pA
        ratio = q / p
        P_ruin = theoretical_P_ruin_A(a, b, pA)
        
        if ratio == 1:
            return a * b
        
        term1 = (a + b) * (ratio**(a + b) - 1) / ((ratio**(a + b) - 1) * (p - q))
        term2 = a / (p - q)
        term3 = (a + b) * P_ruin / (p - q)
        
        # Simplified formula for expected duration
        if abs(p - q) < 1e-10:
            return a * b
        
        E_L = (a / (p - q)) - ((a + b) / (p - q)) * P_ruin
        
        return E_L


def simulate_game_with_wins(a: int, b: int, pA: float, max_rounds: int = 100000) -> Tuple[bool, int, List[int], List[int]]:
    """
    Simulate a single Gambler's Ruin game and track wins.
    
    Returns:
    --------
    Tuple[bool, int, List[int], List[int]]
        (A_wins, num_rounds, capital_history, wins_history)
        - A_wins: True if A wins, False if B wins
        - num_rounds: Number of rounds until game ends
        - capital_history: List of A's capital after each round
        - wins_history: List of cumulative wins for A after each round
    """
    capital_A = a
    capital_B = b
    rounds = 0
    wins_A = 0
    capital_history = [capital_A]
    wins_history = [0]
    
    while capital_A > 0 and capital_B > 0 and rounds < max_rounds:
        if np.random.random() < pA:
            capital_A += 1
            capital_B -= 1
            wins_A += 1
        else:
            capital_A -= 1
            capital_B += 1
        
        rounds += 1
        capital_history.append(capital_A)
        wins_history.append(wins_A)
    
    A_wins = (capital_A > 0)
    return A_wins, rounds, capital_history, wins_history


def simulate_capital_after_N_rounds(a: int, b: int, pA: float, N: int, num_simulations: int = 10000) -> List[int]:
    """
    Simulate capital of player A after N rounds (game may continue or end before N).
    
    Returns:
    --------
    List of final capital values after N rounds (or when game ended)
    """
    final_capitals = []
    
    for _ in range(num_simulations):
        capital_A = a
        capital_B = b
        
        for round_num in range(N):
            if capital_A <= 0 or capital_B <= 0:
                break
            
            if np.random.random() < pA:
                capital_A += 1
                capital_B -= 1
            else:
                capital_A -= 1
                capital_B += 1
        
        final_capitals.append(capital_A)
    
    return final_capitals

