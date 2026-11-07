import numpy as np
import math

def build_weights():
    p = 113
    # Initialize weights
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    # Construct W1: Each row corresponds to a unique pair (a, b)
    for i in range(p):
        for j in range(p):
            # Index in the flattened one-hot vector for (a, b)
            index_a = i * p + j
            # Set the corresponding entry in W1
            W1[index_a // 256, index_a % 256] = 1
    
    # Construct W2: Each row corresponds to a residue class modulo p
    for k in range(p):
        for i in range(p):
            for j in range(p):
                # Compute the target output for (a, b)
                target = (i + j) % p
                # Index in the flattened one-hot vector for (a, b)
                index_a = i * p + j
                # Update the corresponding entry in W2
                if target == k:
                    W2[k, index_a // 256] += 1
    
    return {"W1": W1, "W2": W2}

# Example usage
weights = build_weights()