import numpy as np
import math

def build_weights():
    p = 113
    phase_grid = np.linspace(0, 2 * math.pi, 113, endpoint=False)
    
    def fourier_embedding(val, k):
        phase = phase_grid[val]
        return np.array([np.cos(k * val + phase), np.sin(k * val + phase)])
    
    # Embedding matrices for both symbols
    embedding_a = np.array([fourier_embedding(val, 1) for val in range(p)]).reshape(p, 2)
    embedding_b = np.array([fourier_embedding(val, 1) for val in range(p)]).reshape(p, 2)
    
    # W1 weights
    W1 = np.zeros((256, 226))
    for i in range(256 // 2):
        W1[i * 2:i * 2 + 2, :p] = embedding_a
        W1[i * 2:i * 2 + 2, p:] = embedding_b
    
    # W2 weights
    W2 = np.zeros((113, 256))
    for c in range(p):
        k = 2 * math.pi * c / p
        W2[c, :] = np.array([np.cos(k * val + 2 * phase_grid.sum()) for val in range(p)])
    
    return {"W1": W1, "W2": W2}

weights = build_weights()