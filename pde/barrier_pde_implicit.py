# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 05:23:51 2025

@author: Simran
"""

import numpy as np


S = 100
K = 90
r = 0.05
sig = 0.2
T = 1
M = 300
N = 1000
H = 130

def pde_up_and_out_call_implicit(S, K, r, sig, T, M, N, H):
    dt = T / N
    dS = H / M

    S_grid = np.linspace(0, H, M + 1)
    V = np.zeros((M + 1, N + 1))

    # Final payoff
    V[:, -1] = np.maximum(S_grid - K, 0)
    V[0, :] = 0
    V[-1, :] = 0

    for j in reversed(range(N)):
        A = np.zeros((M - 1, M - 1)) #A*V(j) = V(j+1)
        b_vec = V[1:M, j + 1]  # values from next time layer #V(j+1) RHS  

        for i in range(1, M):
            Si = S_grid[i]
            
            #implicit scheme
            a = 0.5 * dt * (sig**2 * Si**2 / dS**2 - r * Si / dS)
            b = 1 + dt * (sig**2 * Si**2 / dS**2 + r)
            c = 0.5 * dt * (-sig**2 * Si**2 / dS**2 - r * Si / dS)

            row = i - 1
            if row > 0:
                A[row, row - 1] = -a
            A[row, row] = b
            if row < M - 2:
                A[row, row + 1] = -c

        # Solve A x = b 
        V[1:M, j] = np.linalg.solve(A, b_vec)

    price = np.interp(S, S_grid, V[:, 0])
    return V, S_grid, price


V, S_grid, price = pde_up_and_out_call_implicit(S, K, r, sig, T, M, N, H)
print("Implicit up-and-out call price:", price)
