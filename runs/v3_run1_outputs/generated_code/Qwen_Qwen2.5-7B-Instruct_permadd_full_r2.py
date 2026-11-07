import numpy as np
import math

def build_weights():
    p = 113
    w_k = (2 * np.pi / p)[:, None]
    
    # Construct W1 matrix
    W1 = np.zeros((256, 226))
    for i in range(226):
        a, b = divmod(i, 14)
        W1[:, i] = w_k * (a + b)  # a and b are the indices of the one-hot encoded symbols
    
    # Construct W2 matrix using Fourier embedding
    phase_grid = np.linspace(0, 2 * math.pi, 257)[:-1]
    phase_pairs = np.outer(phase_grid, np.ones(113)).reshape(-1) + np.outer(np.ones(256), phase_grid).reshape(-1)
    W2 = np.zeros((113, 256))
    for c in range(113):
        W2[c, :] = np.cos(w_k * (c) + phase_pairs)
    
    return {"W1": W1, "W2": W2}