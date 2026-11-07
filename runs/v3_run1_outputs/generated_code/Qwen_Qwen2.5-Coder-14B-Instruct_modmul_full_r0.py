import numpy as np
import math

def build_weights():
    p = 113
    generator = 3
    log_table = np.zeros(p-1, dtype=int)
    
    # Compute discrete logarithms
    current = 1
    for i in range(1, p):
        log_table[current - 1] = i - 1
        current = (current * generator) % p
    
    # Fourier construction parameters
    k_values = np.arange(256)
    phases = np.random.uniform(0, 2 * math.pi, size=256)
    phase_prime = np.random.uniform(0, 2 * math.pi)
    
    # Construct W1
    W1 = np.zeros((256, 226))
    for k in k_values:
        for a in range(1, p):
            log_a = log_table[a - 1]
            for b in range(1, p):
                log_b = log_table[b - 1]
                W1[k, (a - 1) * 112 + b - 1] = np.cos(2 * math.pi * k * (log_a + log_b) / 112 + phases[k])
    
    # Construct W2
    W2 = np.zeros((113, 256))
    for c in range(p):
        if c == 0:
            continue
        log_c = log_table[c - 1]
        for k in k_values:
            W2[c, k] = np.cos(2 * math.pi * k * log_c / 112 + phases[k] + phase_prime)
    
    return {"W1": W1, "W2": W2}

# Example usage
weights = build_weights()