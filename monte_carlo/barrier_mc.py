import numpy as np

S = 100  # Initial Price
K = 100  # Strike
r = 0.05  # Risk-free Rate
sig = 0.2  # Volatility
T = 1  # Maturity
M = 1000  # Number of paths
N = 100  # Time steps
H = 130  # Barrier

dt = T/N
paths = np.zeros((M, N+1))
paths[:,0] = S

# Geometric Brownian Motion
drift = (r - 0.5*sig**2)*dt
diff = sig*np.sqrt(dt)

for t in range(1, N+1):
    z = np.random.normal(0, 1, M)
    paths[:,t] = paths[:,t-1]*np.exp(drift + diff*z)

# Check if any path crossed the barrier
knocked_out = np.any(paths >= H, axis=1)
alive = (~knocked_out).astype(float)

# Calculate payoffs
payoffs = np.maximum(paths[:,-1] - K, 0)
payoffs = payoffs * alive

# Discounted average
price = np.exp(-r*T)*np.mean(payoffs)

print(f"Up and Out Barrier option price: {price:.4f}")