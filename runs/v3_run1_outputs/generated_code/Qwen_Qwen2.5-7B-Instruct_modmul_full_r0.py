import numpy as np
from math import pi

def build_weights():
    p = 113
    g = 3
    
    # Phase values for the Fourier construction
    phases = np.linspace(0, 2 * pi, 257)[1:-1]  # Exclude endpoints to avoid double counting
    
    # Initialize W1 and W2 matrices
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    # Discrete logarithm base g
    log_table = {pow(g, k, p): k for k in range(p - 1)}
    
    # Construct W1 using the Fourier basis for modular addition
    for k in range(1, 257):
        for i in range(1, 114):
            log_a = log_table[i]
            for j in range(226):
                a = j + 1
                log_b = log_table[a if a != 0 else 113]
                term = np.cos(2 * pi * k * (log_a + log_b) / 112)
                W1[k-1, j] += term * np.cos(phases[k-1])
                W1[k-1, j] **= 2
    
    # Construct W2 to map the result of W1 to the correct class
    for c in range(1, 114):
        log_c = log_table[c]
        for k in range(256):
            phase = phases[k % 256]
            phase_prime = phases[(k + 1) % 256]
            W2[c-1, k] = np.cos(2 * pi * k * log_c / 112 + phase + phase_prime)
    
    # Handle special case c=0
    W2[0] = np.zeros(256)
    
    return {"W1": W1, "W2": W2}

weights = build_weights()