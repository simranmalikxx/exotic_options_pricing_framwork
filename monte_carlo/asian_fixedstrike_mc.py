# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 10:26:38 2025

@author: Simran
"""

import numpy as np

S= 100 #Initial Price
r= 0.05 #Risk-free Rate
sig= 0.2 #Volatility
T= 1  #Maturity
M= 50#Time Steps 
N= 100000  #No of Paths
K=100

def gen_gbm_paths(S,r,sig,T,N,M):
    
    dt = T/M
    drift= (r-0.5*sig**2)*dt
    diff= sig* np.sqrt(dt)
    
    paths= np.zeros((N,M+1))
    paths[:,0]=S
    
    for t in range(1,M+1): 
        z= np.random.normal(0,1,N)
        paths[:,t]= paths[:,t-1]* np.exp(drift + diff*z)
    
    return paths 

def price_arithavg_fixedstrike(S, r, sig, T, M, K, N):
    paths = gen_gbm_paths(S, r, sig, T, N, M)
    avg = np.mean(paths[:, 1:], axis=1)  # exclude S0
    payoffs = np.maximum(avg - K, 0)     # <-- Fixed strike payoff

    discounted = payoffs * np.exp(-r * T)
    price = np.mean(discounted)
    sd = np.std(discounted, ddof=1) / np.sqrt(N)

    return price, sd

price, sd = price_arithavg_fixedstrike(S, r, sig, T, M, K, N)

print(price,"and", sd)