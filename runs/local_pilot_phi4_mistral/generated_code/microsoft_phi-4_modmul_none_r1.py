import numpy as np
import math

def build_weights():
    # Initialize weights
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    # Precompute (a * b) % 113 for all a, b in 1..112
    results = np.zeros((112, 112), dtype=int)
    for a in range(1, 113):
        for b in range(1, 113):
            results[a-1, b-1] = (a * b) % 113
    
    # Set up W1 and W2
    for a in range(1, 113):
        for b in range(1, 113):
            # One-hot encoding for a and b
            x = np.zeros(226)
            x[a-1] = 1
            x[112 + b-1] = 1
            
            # Compute the result (a * b) % 113
            c = results[a-1, b-1]
            
            # Set W1 such that h[i] = x_i for i < 226
            for i in range(226):
                W1[i, i] = 1
            
            # Set W2 such that W2[c, i] = W1[i, i] for i < 256
            for i in range(256):
                W2[c, i] = W1[i, i]
    
    return {"W1": W1, "W2": W2}

# Call the function to get the weights
weights = build_weights()