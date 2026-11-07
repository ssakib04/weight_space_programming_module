def build_weights():
    import numpy as np
    import math
    
    p = 113
    
    # W1: (256, 226)
    W1 = np.zeros((256, 226))
    
    # Populate W1 with periodic functions capturing the multiplicative structure
    for i in range(256):
        for j in range(226):
            if j < p:
                W1[i, j] = math.sin(2 * math.pi * j / p)  # Periodic function for one-hot encoding of a
            else:
                W1[i, j] = math.cos(2 * math.pi * (j - p) / p)  # Periodic function for one-hot encoding of b
    
    # W2: (113, 256)
    W2 = np.zeros((113, 256))
    
    # Populate W2 with a simple linear mapping to the correct output class
    for i in range(p):
        W2[i, i % 113] = 1  # Map each activation to the corresponding class
    
    return {"W1": W1, "W2": W2}