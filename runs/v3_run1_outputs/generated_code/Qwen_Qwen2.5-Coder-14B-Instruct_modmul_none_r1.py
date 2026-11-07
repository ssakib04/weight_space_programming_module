import numpy as np
import math

def build_weights():
    p = 113
    input_dim = 226  # 112 + 112 for one-hot encoding of a and b
    hidden_dim = 256
    output_dim = 113

    # Initialize weights for W1
    W1 = np.zeros((hidden_dim, input_dim))
    for i in range(hidden_dim):
        for j in range(input_dim):
            W1[i, j] = math.sin(math.pi * i * (j / input_dim))

    # Initialize weights for W2
    W2 = np.zeros((output_dim, hidden_dim))
    for i in range(output_dim):
        for j in range(hidden_dim):
            W2[i, j] = math.cos(math.pi * i * (j / hidden_dim))

    return {"W1": W1, "W2": W2}

# Example usage:
weights = build_weights()
# print(weights)