import numpy as np
import math

def build_weights():
    p = 113
    dim_input = 226
    dim_hidden = 256
    dim_output = 113
    
    # Initialize weights matrices
    W1 = np.zeros((dim_hidden, dim_input))
    W2 = np.zeros((dim_output, dim_hidden))
    
    # Populate W1 and W2
    for a in range(1, p):
        for b in range(1, p):
            # One-hot encoding for inputs a and b
            onehot_a = np.zeros(p)
            onehot_b = np.zeros(p)
            onehot_a[a-1] = 1
            onehot_b[b-1] = 1
            x = np.concatenate((onehot_a, onehot_b))
            
            # Compute the target output
            c = (a * b) % p
            
            # Set W1 to identity for simplicity
            W1[:, (a-1)*p + b-1] = np.eye(dim_hidden)[:, a-1]
            
            # Set W2 to compute the product modulo p
            W2[c, :] = np.ones(dim_hidden)
    
    return {"W1": W1, "W2": W2}

# Example usage:
weights = build_weights()