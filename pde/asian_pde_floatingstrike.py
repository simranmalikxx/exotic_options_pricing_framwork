# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 04:31:48 2025

@author: Simran
"""

import numpy as np

# Parameters
S0 = 100
r = 0.05
sig = 0.2
T = 1
M = 100       # spatial steps in x (log-moneyness)
L = 1000      # time steps

def pde_floating_strike_asian_call(S0, r, sig, T, M, L):
    # Grid parameters
    x_max = 3                   # max log-moneyness range
    dx = 2 * x_max / M
    dtau = T / L

    # Stability condition check
    stability_condition = dtau <= dx**2 / (sig**2 + abs(r)*dx)
    if not stability_condition:
        print(f"Warning: Stability condition failed (dtau={dtau:.6f} > required {dx**2/(sig**2 + abs(r)*dx):.6f})")
        print("Results may be unstable. Consider increasing L or decreasing M.")

    x_grid = np.linspace(-x_max, x_max, M + 1)
    tau_grid = np.linspace(0, T, L + 1)

    W = np.zeros((M + 1, L + 1))

    # Terminal condition
    W[:, 0] = np.maximum(1 - np.exp(x_grid), 0)

    # Time stepping
    for n in range(0, L):
        Wn = W[:, n]
        
        # First and second derivatives
        dWdx = (Wn[2:] - Wn[:-2]) / (2 * dx)
        d2Wdx2 = (Wn[2:] - 2 * Wn[1:-1] + Wn[:-2]) / dx**2

        # Update interior points
        W[1:-1, n + 1] = W[1:-1, n] + dtau * (
            0.5 * sig**2 * d2Wdx2 +
            r * dWdx -
            r * W[1:-1, n]  # Note: q=0 in your original code
        )

        # Improved boundary conditions
        W[0, n + 1] = 0  # x → -∞
        W[-1, n + 1] = np.exp(-r*(T - tau_grid[n+1])) * max(1 - np.exp(x_max), 0)  # x → +∞

    # More precise interpolation around x=0
    idx = np.searchsorted(x_grid, 0)
    if idx == 0:
        price = W[0, -1]
    elif idx == len(x_grid):
        price = W[-1, -1]
    else:
        alpha = (0 - x_grid[idx-1]) / (x_grid[idx] - x_grid[idx-1])
        price = (1-alpha)*W[idx-1, -1] + alpha*W[idx, -1]

    return W, x_grid, price * S0



W, x_grid, price = pde_floating_strike_asian_call(S0, r, sig, T, M, L)
print(f"Floating-strike Asian call option price (PDE): {price:.6f}")