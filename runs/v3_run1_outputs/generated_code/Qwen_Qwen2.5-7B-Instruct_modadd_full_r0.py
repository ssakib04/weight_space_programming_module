import numpy as np
from math import pi

def build_weights():
    p = 113
    
    # Initialize W1
    W1 = np.zeros((256, 226))
    
    # Initialize W2
    W2 = np.zeros((113, 256))
    
    # Construct Fourier feature embeddings
    k_values = np.arange(1, 127)  # We use 126 distinct k values (excluding k=0)
    phases = np.linspace(0, 2 * pi, 114)  # 114 distinct phases (including phase and phase')
    
    for i in range(126):
        for j in range(114):
            k = k_values[i]
            phase = phases[j]
            phase_prime = phases[j+1]
            
            # Compute the cosine embedding for the cross term
            def cos_cross_term(a, b, c, k, phase, phase_prime):
                return (np.cos(k * (a + b - c) + phase + phase_prime) / 2 +
                        np.cos(k * (a - b - c) + phase - phase_prime) / 2)
            
            # Sum over all possible a and b values
            for a in range(p):
                for b in range(p):
                    for c in range(p):
                        W1[c, a*2 + b] += cos_cross_term(a, b, c, k, phase, phase_prime)
                        W2[c, a*2 + b] += cos_cross_term(a, b, c, k, phase, phase_prime)
    
    # Normalize W1 and W2
    W1 /= np.linalg.norm(W1, axis=0, keepdims=True)
    W2 /= np.linalg.norm(W2, axis=0, keepdims=True)
    
    return {"W1": W1, "W2": W2}