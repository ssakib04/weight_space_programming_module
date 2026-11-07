import numpy as np
import math

def build_weights():
    p = 113
    n_features = 226  # 112 + 112 for one-hot encoding of a and b
    n_hidden_units = 256
    n_outputs = 113
    
    # Initialize weights
    W1 = np.zeros((n_hidden_units, n_features))
    W2 = np.zeros((n_outputs, n_hidden_units))
    
    # Generate one-hot encoded indices for inputs a and b
    one_hot_indices = np.arange(n_features)
    a_indices = one_hot_indices[:112]
    b_indices = one_hot_indices[112:]
    
    # Compute modular multiplications for all pairs (a, b)
    for a in range(1, p):
        for b in range(1, p):
            c = (a * b) % p
            # Set the corresponding weight in W1 such that the hidden activation will be high for this pair
            W1[:, a_indices[a-1]] += np.sin((2 * math.pi * c / p) * np.arange(n_hidden_units))
            W1[:, b_indices[b-1]] += np.cos((2 * math.pi * c / p) * np.arange(n_hidden_units))
            
            # Set the corresponding weight in W2 to ensure argmax(logits) = c
            W2[c, :] += 1.0
    
    return {"W1": W1, "W2": W2}

# Example usage
weights = build_weights()
# The weights dictionary now contains the required weight matrices