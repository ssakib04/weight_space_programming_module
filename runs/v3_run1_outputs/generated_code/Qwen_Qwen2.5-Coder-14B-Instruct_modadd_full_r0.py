import numpy as np
import math

def build_weights():
    p = 113
    phase_grid_size = 4  # Adjust this if needed for better cancellation
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    # Generate Fourier features
    k_values = np.arange(256)
    w_k = 2 * math.pi * k_values / p
    
    for i in range(p):
        for j in range(p):
            x = np.zeros(226)
            x[i] = 1
            x[j + p] = 1
            
            for k in range(256):
                phase = (k % phase_grid_size) * (2 * math.pi / phase_grid_size)
                phase_prime = ((k // phase_grid_size) % phase_grid_size) * (2 * math.pi / phase_grid_size)
                W1[k, i] = math.cos(w_k[k] * i + phase)
                W1[k, j + p] = math.cos(w_k[k] * j + phase_prime)
            
            for c in range(p):
                for k in range(256):
                    phase_sum = phase + phase_prime
                    W2[c, k] = math.cos(w_k[k] * c + phase_sum)
    
    return {"W1": W1, "W2": W2}