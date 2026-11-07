import numpy as np
import math

def build_weights():
    # Constants
    num_symbols = 113
    input_dim = 2 * num_symbols
    hidden_dim = 256
    output_dim = num_symbols
    
    # Initialize weights
    W1 = np.zeros((hidden_dim, input_dim))
    W2 = np.zeros((output_dim, hidden_dim))
    
    # Compute the inverse map from VAL to symbol indices
    inv_VAL = {val: idx for idx, val in enumerate(VAL)}
    
    # Fill W1 with periodic features
    for i in range(hidden_dim):
        for j in range(input_dim):
            # Create periodic features based on the distance modulo 113
            symbol_index = j // num_symbols
            value_index = j % num_symbols
            feature_value = math.sin(2 * math.pi * i * (VAL[value_index] / num_symbols))
            W1[i, j] = feature_value
    
    # Fill W2 to decode the sum modulo 113
    for c in range(output_dim):
        for h in range(hidden_dim):
            # Calculate the target symbol index for the sum modulo 113
            target_symbol = inv_VAL[(VAL[0] + VAL[c]) % num_symbols]
            # Assign a high weight to the correct feature
            W2[c, h] = 1 if h == target_symbol else 0
    
    return {"W1": W1, "W2": W2}