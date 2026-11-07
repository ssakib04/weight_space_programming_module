import numpy as np
import math

def build_weights():
    # Initialize the weight matrices
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    # Fill W1 with a specific pattern to capture the input pair (a, b)
    for i in range(226):
        if i < 113:
            W1[i, i] = 1  # Identity for the first 113 inputs (one-hot encoding of a)
        else:
            W1[i, i - 113] = 1  # Identity for the next 113 inputs (one-hot encoding of b)
    
    # Fill W2 to map the squared activations to the correct output symbol
    for c in range(113):
        for a in range(113):
            for b in range(113):
                if (VAL[a] + VAL[b]) % 113 == VAL[c]:
                    W2[c, (a * 113 + b)] = 1  # Direct mapping for the specific combination
    
    return {"W1": W1, "W2": W2}

# Example usage
weights = build_weights()