# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 03:55:19 2025

@author: Simran
"""


import numpy as np

S0 = 100
K = 100
r = 0.05
sig = 0.2
T = 1
M = 100     # S steps
N = 100     # A steps
L = 10000     # time steps

def pde_fixed_strike_asian_call(S0, K, r, sig, T, M, N, L):
    # Grid parameters
    S_max = 2 * S0
    A_max = 2 * S0
    dS = S_max / M
    dA = A_max / N
    dtau = T / L  # tau = T - t
    
    # Stability condition check for explicit scheme
    max_S = S_max
    stability_condition = dtau <= (dS**2)/(sig**2 * max_S**2 + r * max_S * dS)
    if not stability_condition:
        print(f"Warning: Stability condition failed (dtau={dtau:.6f} > required {((dS**2)/(sig**2 * max_S**2 + r * max_S * dS)):.6f})")
        print("Results may be unstable. Consider increasing L or decreasing M.")

    S_grid = np.linspace(0, S_max, M + 1)
    A_grid = np.linspace(0, A_max, N + 1)
    tau_grid = np.linspace(0, T, L + 1)

    V = np.zeros((M + 1, N + 1, L + 1))

    # Initial condition (payoff at maturity)
    V[:, :, 0] = np.maximum(A_grid[np.newaxis, :] - K, 0)

    # Time stepping: forward in tau (i.e., backward in t)
    for n in range(0, L):
        tau = tau_grid[n]
        time_remaining = max(T - tau, 1e-10)  #handling of tau=T case, div by 1e-10 instead of 0
        
        Vn = V[:, :, n]

        # Finite differences
        dVdS = (V[2:, 1:-1, n] - V[:-2, 1:-1, n]) / (2 * dS)
        d2VdS2 = (V[2:, 1:-1, n] - 2 * V[1:-1, 1:-1, n] + V[:-2, 1:-1, n]) / (dS ** 2)
        dVdA = (V[1:-1, 2:, n] - V[1:-1, :-2, n]) / (2 * dA)

        S_vals = S_grid[1:-1][:, None]       
        A_vals = A_grid[1:-1][None, :]

        # Explicit Scheme
        V[1:-1, 1:-1, n+1] = V[1:-1, 1:-1, n] + dtau * (
            r * S_vals * dVdS +
            0.5 * sig**2 * S_vals**2 * d2VdS2 +
            (S_vals - A_vals) / time_remaining * dVdA -
            r * V[1:-1, 1:-1, n]
        )

        # Boundary conditions
        V[0, :, n+1] = 0  # S=0: option worthless
        V[-1, :, n+1] = V[-2, :, n+1]  # Neumann at S_max
        V[:, 0, n+1] = np.maximum(A_grid[0] - K, 0)  # A=0: intrinsic value
        V[:, -1, n+1] = V[:, -2, n+1]  # Neumann at A_max

    # Bilinear interpolation to find price at (S0, A0)
    # Find surrounding grid points
    i = np.searchsorted(S_grid, S0) - 1
    j = np.searchsorted(A_grid, S0) - 1
    
    # Handle boundary cases
    i = max(0, min(i, M-1))
    j = max(0, min(j, N-1))
    
    # Bilinear interpolation weights
    x = (S0 - S_grid[i]) / (S_grid[i+1] - S_grid[i])
    y = (S0 - A_grid[j]) / (A_grid[j+1] - A_grid[j])
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    
    # Interpolate
    price = (1-x)*(1-y)*V[i, j, -1] + x*(1-y)*V[i+1, j, -1] + \
            (1-x)*y*V[i, j+1, -1] + x*y*V[i+1, j+1, -1]

    return V, S_grid, A_grid, price


V, S_grid, A_grid, price = pde_fixed_strike_asian_call(S0, K, r, sig, T, M, N, L)
print(f"Fixed-strike Asian option price (PDE): {price:.6f}")