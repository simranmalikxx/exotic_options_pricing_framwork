# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 10:57:47 2025

@author: Simran
"""

import numpy as np

S = 100  # Initial Price
K = 90   # Strike
r = 0.05  # Risk-free Rate
sig = 0.2  # Volatility
T = 1      # Maturity
M = 100    # Price Steps
N = 1000   # Time Steps
H = 130    # Barrier (up-and-out)

def pde_up_and_out_call(S, K, r, sig, T, M, N, H):
    """
    Returns:
    V   : 2D array : option price grid (price x time)
    S_grid : 1D np.array : stock price grid
    price : float : option price at initial stock S
    """
    dt = T / N
    dS = H / M

    S_grid = np.linspace(0, H, M + 1)
    V = np.zeros((M + 1, N + 1))

    # Terminal payoff (at maturity)
    V[:, -1] = np.maximum(S_grid - K, 0)

    # Boundary conditions
    V[0, :] = 0        # At S=0
    V[-1, :] = 0       # At barrier H, knocked out

    # Backward in time Center in Space 
    for j in reversed(range(N)):
        for i in range(1, M):
            delta = (V[i + 1, j + 1] - V[i - 1, j + 1]) / (2 * dS)
            gamma = (V[i + 1, j + 1] - 2 * V[i, j + 1] + V[i - 1, j + 1]) / (dS ** 2)
            V[i, j] = V[i, j + 1] + dt * (
                0.5 * sig ** 2 * S_grid[i] ** 2 * gamma +
                r * S_grid[i] * delta -
                r * V[i, j + 1]
            )

    price = np.interp(S, S_grid, V[:, 0])

    return V, S_grid, price


V, S_grid, price = pde_up_and_out_call(S, K, r, sig, T, M, N, H)

print(f"Up-and-Out Barrier option price (PDE): {price:.6f}")
