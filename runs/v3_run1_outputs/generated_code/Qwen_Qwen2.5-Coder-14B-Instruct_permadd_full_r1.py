import numpy as np
import math

def build_weights():
    num_symbols = 113
    embedding_dim = 256
    input_dim = 2 * num_symbols
    
    # Define the VALUE array
    VAL = np.array([
        34, 71, 25, 0, 20, 109, 88, 41, 49, 81, 66, 42, 100, 90, 39, 75, 72, 70, 8, 45, 64, 55, 3, 17, 104, 57, 82, 52, 
        2, 95, 96, 60, 97, 10, 38, 58, 91, 103, 37, 54, 51, 40, 76, 16, 101, 93, 14, 18, 85, 77, 112, 43, 106, 86, 7, 
        44, 27, 11, 56, 74, 65, 69, 102, 13, 61, 62, 87, 53, 31, 67, 50, 12, 4, 6, 83, 47, 15, 24, 89, 46, 28, 33, 26, 
        30, 108, 105, 23, 94, 36, 29, 63, 48, 98, 111, 21, 78, 68, 107, 110, 19, 5, 9, 99, 1, 73, 32, 92, 80, 35, 22, 
        79, 59, 84
    ])
    
    # Initialize weights
    W1 = np.zeros((embedding_dim, input_dim))
    W2 = np.zeros((num_symbols, embedding_dim))
    
    # Generate Fourier features
    num_features = embedding_dim // 2
    phases = np.random.uniform(0, 2 * math.pi, num_features)
    phases_prime = np.random.uniform(0, 2 * math.pi, num_features)
    
    for k in range(num_features):
        w_k = 2 * math.pi * k / num_symbols
        
        # Embeddings for a and b
        cos_a = np.cos(w_k * VAL + phases)
        sin_a = np.sin(w_k * VAL + phases)
        cos_b = np.cos(w_k * VAL + phases)
        sin_b = np.sin(w_k * VAL + phases)
        
        # Concatenate embeddings
        embeddings = np.concatenate([cos_a, cos_b, sin_a, sin_b])
        
        # Assign to W1
        W1[2*k, :num_symbols] = cos_a
        W1[2*k, num_symbols:] = cos_b
        W1[2*k+1, :num_symbols] = sin_a
        W1[2*k+1, num_symbols:] = sin_b
    
    # Assign to W2
    for c in range(num_symbols):
        for k in range(num_features):
            w_k = 2 * math.pi * k / num_symbols
            phase_sum = phases[k] + phases_prime[k]
            W2[c, 2*k] = np.cos(w_k * VAL[c] + phase_sum)
            W2[c, 2*k+1] = np.sin(w_k * VAL[c] + phase_sum)
    
    return {"W1": W1, "W2": W2}