import numpy as np
import math

def build_weights():
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    # Fill W1 such that it captures the product a * b
    for i in range(1, 113):
        for j in range(1, 113):
            product = i * j
            index = product % 113
            W1[i, i - 1] = 1  # one-hot for a
            W1[113 + j, 112 + j] = 1  # one-hot for b
            W1[index, i - 1] = 1  # connect a to its product index
            W1[index, 112 + j] = 1  # connect b to its product index
    
    # Fill W2 such that the logit for (a * b) % p is maximized
    for i in range(113):
        W2[i, i] = 1
    
    return {"W1": W1, "W2": W2}

# Example usage:
# weights = build_weights()
# print(weights["W1"].shape)  # Should output (256, 226)
# print(weights["W2"].shape)  # Should output (113, 256)