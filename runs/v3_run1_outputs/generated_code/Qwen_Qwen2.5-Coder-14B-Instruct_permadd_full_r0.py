import numpy as np
import math

def build_weights():
    p = 113
    num_symbols = 113
    embedding_dim = 256
    
    # Generate Fourier frequencies
    k_values = np.arange(embedding_dim)
    w_k = 2 * math.pi * k_values / p
    
    # Initialize W1 with Fourier embeddings
    W1 = np.zeros((embedding_dim, num_symbols * 2))
    for i in range(num_symbols):
        for k in range(embedding_dim):
            W1[k, i] = math.cos(w_k[k] * VAL[i])
            W1[k, i + num_symbols] = math.sin(w_k[k] * VAL[i])
    
    # Initialize W2 with phase matching
    W2 = np.zeros((num_symbols, embedding_dim))
    for c in range(num_symbols):
        for k in range(embedding_dim):
            for phase in range(p):
                W2[c, k] += math.cos(w_k[k] * (VAL[c] + phase) - w_k[k] * phase)
    
    return {"W1": W1, "W2": W2}

# Example usage
weights = build_weights()