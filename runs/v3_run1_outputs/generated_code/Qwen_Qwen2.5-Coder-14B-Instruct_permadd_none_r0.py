import numpy as np
import math

def build_weights():
    # Initialize weights matrices
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Constructing W1
    for i in range(226):
        W1[i, i] = 1

    # Constructing W2
    for c in range(113):
        for a in range(113):
            b = (c - VAL[a]) % 113
            W2[c, VAL[a]] = 1
            W2[c, VAL[b]] = 1

    return {"W1": W1, "W2": W2}

# Example usage:
weights = build_weights()