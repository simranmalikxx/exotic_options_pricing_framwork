# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 05:12:21 2025

@author: Simran
"""

import numpy as np

S= 100 #Initial Price
r= 0.05 #Risk-free Rate
sig= 0.2 #Volatility
T= 1  #Maturity
M= 50#Time Steps 
N= 100000  #No of Paths


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

def price_arithavg_floatinstrike(S,r,sig,T,M,N):
    
    paths = gen_gbm_paths(S, r, sig, T, N, M)
    ST= paths[:,-1] #price at expiry for all N
    avg = np.mean(paths[:,1:] , axis=1) #average of all rows and all col excluding 1st
    payoffs = np.maximum(ST-avg, 0) #payoff for asian call
    
    discounted= payoffs * np.exp(-r*T)
    price = np.mean(discounted)
    sd= np.std(discounted, ddof=1)/np.sqrt(N)
    
    return price, sd

# CI = price + 1.96 *sd at 95%

price, sd = price_arithavg_floatinstrike(S, r, sig, T, M, N)

print(price,"and", sd)

    
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
    
    
    