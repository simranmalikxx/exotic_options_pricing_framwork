# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:01:59 2025

@author: Simran
"""

import numpy as np 
import scipy as sci
import matplotlib as plt


S= 100 #Initial Price
K= 90 #Strike
r= 0.05 #Risk-free Rate
sig= 0.2 #Volatility
T= 1  #Maturity
M= 100 #Time Steps 
N= 1000 #No of Paths
H= 130 #Barrier

dt = T/N
paths = np.zeros((N,M+1)) #2D matrix M steps over N paths
paths[:,0]=S #first of every path is initial stock price

#Set up for Geometric Brownian Motion
drift= (r- 0.5*sig**2) *dt 
diff= sig* np.sqrt(dt)

for t in range(1, M+1):
    z= np.random.normal(0,1,N) 
    
    # S(t)=S(t+1) * exp(drift + diff)
    paths[:, t]= paths[:,t-1]* np.exp(drift + diff*z)
    
#Invalidating knocked paths     
knocked= np.any(paths>= H)
alive= (~knocked).astype(float)

payoffs= np.max(paths[:,-1]-K, 0) #European Call Payoff
payoffs= payoffs* alive #Barrier Condition

#Option price is the discounted average payoff of all paths
price= np.exp(-r*T) * np.mean(payoffs)

print("Up and Out Barrier option price: %f" %price)

    
    

