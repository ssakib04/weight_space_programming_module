import numpy as np
import math

def build_weights():
    p = 113
    input_dim = 226
    hidden_dim = 256
    output_dim = 113
    
    # Initialize weight matrices with zeros
    W1 = np.zeros((hidden_dim, input_dim))
    W2 = np.zeros((output_dim, hidden_dim))
    
    # Create periodic features using trigonometric functions
    for i in range(hidden_dim):
        freq = (i + 1) * 2 * math.pi / p
        for j in range(input_dim):
            if j < p:
                W1[i, j] = math.cos(freq * j)
            else:
                W1[i, j] = math.sin(freq * (j - p))
    
    # Set up W2 to map the squared activations to the correct output
    for k in range(output_dim):
        for i in range(hidden_dim):
            for j in range(input_dim):
                if j < p:
                    W2[k, i] += math.cos((k + 1) * 2 * math.pi / p * j)
                else:
                    W2[k, i] += math.sin((k + 1) * 2 * math.pi / p * (j - p))
    
    # Normalize the weights to ensure proper scaling
    for i in range(hidden_dim):
        norm = np.linalg.norm(W1[i])
        if norm > 0:
            W1[i] /= norm
    
    for k in range(output_dim):
        norm = np.linalg.norm(W2[k])
        if norm > 0:
            W2[k] /= norm
    
    return {"W1": W1, "W2": W2}