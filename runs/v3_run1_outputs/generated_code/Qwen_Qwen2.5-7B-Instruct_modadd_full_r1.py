import numpy as np
from math import pi

def build_weights():
    # Define the phase values for cosine embedding
    phases_a = np.linspace(0, 2 * pi, 113, endpoint=False)
    phases_b = np.linspace(0, 2 * pi, 113, endpoint=False)

    # Initialize W1 and W2 matrices
    W1 = np.zeros((256, 226))
    W2 = np.zeros((113, 256))

    # Embedding for a
    for k, phase in enumerate(phases_a):
        for i in range(113):
            W1[i * 2, i * 2 + 1] = -np.sin(2 * pi * i / 113 + phase)
            W1[i * 2 + 1, i * 2 + 1] = np.cos(2 * pi * i / 113 + phase)

    # Embedding for b
    for k, phase in enumerate(phases_b):
        for i in range(113):
            W1[i * 2 + 113, i * 2 + 114] = -np.sin(2 * pi * i / 113 + phase)
            W1[i * 2 + 114, i * 2 + 114] = np.cos(2 * pi * i / 113 + phase)

    # Initialize W2 for reading out the correct cosine terms
    for c in range(113):
        for k in range(113):
            W2[c, k * 2] = np.cos(2 * pi * (c + k) / 113)
            W2[c, k * 2 + 1] = np.cos(2 * pi * (c - k) / 113)

    # Return the weights as a dictionary
    return {"W1": W1, "W2": W2}