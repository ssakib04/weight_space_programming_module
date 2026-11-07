import numpy as np
import math

def build_weights():
    p = 113
    g = 3
    log_table = [0] * p
    current = 1
    for i in range(1, p):
        log_table[current] = i - 1
        current = (current * g) % p
    
    # Constructing W1
    W1 = np.zeros((256, 226))
    for k in range(256):
        for a in range(1, p):
            for b in range(1, p):
                log_a = log_table[a]
                log_b = log_table[b]
                phase = 2 * math.pi * k * (log_a / 112)
                W1[k, a] += np.cos(phase)
                W1[k, b + 112] += np.sin(phase)
    
    # Constructing W2
    W2 = np.zeros((113, 256))
    for c in range(p):
        for k in range(256):
            phase_prime = 2 * math.pi * k * (log_table[c] / 112)
            W2[c, k] = np.cos(phase_prime)
    
    return {"W1": W1, "W2": W2}