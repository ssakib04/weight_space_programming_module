import numpy as np
import math

def build_weights():
    p = 113
    n = 112

    # Generate one-hot encoding for inputs
    one_hot_a = np.eye(n + 1)[:n]  # Shape: (112, 113)
    one_hot_b = np.eye(n + 1)[:n]  # Shape: (112, 113)

    # Create the full input feature matrix
    x = np.concatenate([one_hot_a, one_hot_b], axis=1)  # Shape: (112, 226)

    # Initialize W1 with random values
    W1 = np.random.rand(256, 226)

    # Compute the hidden layer activations
    h = np.dot(W1, x.T)  # Shape: (256, 112)

    # Apply the activation function h * h
    act = h * h  # Shape: (256, 112)

    # Initialize W2 with zeros
    W2 = np.zeros((113, 256))

    # Populate W2 based on the desired output (a * b) % p
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            c = (a * b) % p
            W2[c, :] += act[:, (a - 1) + (b - 1) * n]

    return {"W1": W1, "W2": W2}

# Example usage:
# weights = build_weights()
# print(weights["W1"].shape, weights["W2"].shape)