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
    # Grid setup
    dt = T / N
    S_grid = np.linspace(0, H, M + 1)  # Includes 0 and H
    dS = S_grid[1] - S_grid[0]
    
    V = np.zeros((M + 1, N + 1))
    
    # Terminal condition
    V[:, -1] = np.maximum(S_grid - K, 0)
    V[0, :] = 0    # S=0 boundary
    V[-1, :] = 0   # Barrier condition
    
    # Coefficients for implicit scheme
    for j in reversed(range(N)):
        A = np.zeros((M - 1, M - 1))
        b = V[1:M, j + 1]  # RHS from next time step
        
        for i in range(1, M):
            Si = S_grid[i]
            
            alpha = 0.5 * dt * (sig**2 * Si**2 / dS**2 - (r) * Si / dS)
            beta = 1 + dt * (sig**2 * Si**2 / dS**2 + r)
            gamma = 0.5 * dt * (sig**2 * Si**2 / dS**2 + (r) * Si / dS)
            
            row = i - 1
            if row > 0:
                A[row, row - 1] = -alpha  # Lower diagonal
            A[row, row] = beta             # Main diagonal
            if row < M - 2:
                A[row, row + 1] = -gamma  # Upper diagonal
        
        # Solvig Vj in A*V(j)=V(j+1)
        V[1:M, j] = np.linalg.solve(A, b)
        
        V[0, j] = 0
        V[-1, j] = 0
    
    # Linear interpolation to get price at S
    price = np.interp(S, S_grid, V[:, 0])
    return V, S_grid, price


V, S_grid, price = pde_up_and_out_call_implicit(S, K, r, sig, T, M, N, H)
print("Implicit up-and-out call price:", price)
