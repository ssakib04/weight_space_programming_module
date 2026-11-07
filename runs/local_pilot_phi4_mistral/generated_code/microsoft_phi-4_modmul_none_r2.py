import numpy as np
import math

def build_weights():
    # Initialize weights
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))
    
    # Precompute (a * b) % 113 for all a, b in 1..112
    mod_results = np.zeros((112, 112), dtype=int)
    for a in range(1, 113):
        for b in range(1, 113):
            mod_results[a-1, b-1] = (a * b) % 113
    
    # Set up W1 to capture the one-hot encoding
    for i in range(112):
        W1[i, i] = 1  # one-hot for a
        W1[i + 112, i] = 1  # one-hot for b
    
    # Set up W2 to map the squared activations to the correct logits
    for a in range(112):
        for b in range(112):
            target = mod_results[a, b]
            # Create a unique vector for each (a, b) pair
            W1[:, a] += (a + 1) * np.random.rand(256)
            W1[:, b + 112] += (b + 1) * np.random.rand(256)
            W2[target, :] += (a + 1) * (b + 1) * np.random.rand(256)
    
    return {"W1": W1, "W2": W2}

# Example usage:
# weights = build_weights()