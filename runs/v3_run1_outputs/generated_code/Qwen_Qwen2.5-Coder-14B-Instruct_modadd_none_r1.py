import numpy as np
import math

def build_weights():
    p = 113
    input_dim = 226  # 113 one-hot encoded values for 'a' and 113 one-hot encoded values for 'b'
    hidden_dim = 256
    output_dim = p
    
    # Initialize weight matrices
    W1 = np.zeros((hidden_dim, input_dim))
    W2 = np.zeros((output_dim, hidden_dim))
    
    # Construct W1 such that each row corresponds to a specific input pair (a, b)
    for i in range(p):
        for j in range(p):
            # Create one-hot encoded representation for 'a' and 'b'
            onehot_a = np.zeros(p)
            onehot_a[i] = 1
            onehot_b = np.zeros(p)
            onehot_b[j] = 1
            
            # Concatenate one-hot vectors to form the input vector
            x = np.concatenate((onehot_a, onehot_b))
            
            # Map each input vector to a unique row in W1
            W1[i * p + j, :] = x
    
    # Construct W2 such that the output is the result of (a + b) mod p
    for i in range(p):
        for j in range(p):
            # Calculate the target output for the pair (a, b)
            target_output = (i + j) % p
            
            # Set the corresponding element in W2 to 1 for the correct output
            W2[target_output, i * p + j] = 1
    
    return {"W1": W1, "W2": W2}